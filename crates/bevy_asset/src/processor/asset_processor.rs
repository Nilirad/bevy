pub use super::*;

use crate::{
    io::{
        AssetReader, AssetReaderError, AssetSource, AssetSourceBuilders, AssetSourceEvent,
        AssetSourceId, AssetSources, AssetWriter, AssetWriterError, MissingAssetSourceError,
    },
    meta::{
        get_asset_hash, get_full_asset_hash, AssetAction, AssetActionMinimal, AssetMeta,
        AssetMetaDyn, AssetMetaMinimal, ProcessedInfo, ProcessedInfoMinimal,
    },
    AssetMetaCheck, AssetPath, AssetServer, AssetServerMode, DeserializeMetaError,
    MissingAssetLoaderForExtensionError,
};

use bevy_log::{debug, error, trace, warn};
use bevy_tasks::IoTaskPool;
use bevy_utils::BoxedFuture;
use futures_io::ErrorKind;
use futures_lite::{AsyncReadExt, AsyncWriteExt, StreamExt};

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

/// A "background" asset processor that reads asset values from a source [`AssetSource`] (which corresponds to an [`AssetReader`] / [`AssetWriter`] pair),
/// processes them in some way, and writes them to a destination [`AssetSource`].
///
/// This will create .meta files (a human-editable serialized form of [`AssetMeta`]) in the source [`AssetSource`] for assets that
/// that can be loaded and/or processed. This enables developers to configure how each asset should be loaded and/or processed.
///
/// [`AssetProcessor`] can be run in the background while a Bevy App is running. Changes to assets will be automatically detected and hot-reloaded.
///
/// Assets will only be re-processed if they have been changed. A hash of each asset source is stored in the metadata of the processed version of the
/// asset, which is used to determine if the asset source has actually changed.  
///
/// A [`ProcessorTransactionLog`] is produced, which uses "write-ahead logging" to make the [`AssetProcessor`] crash and failure resistant. If a failed/unfinished
/// transaction from a previous run is detected, the affected asset(s) will be re-processed.
///
/// [`AssetProcessor`] can be cloned. It is backed by an [`Arc`] so clones will share state. Clones can be freely used in parallel.
#[derive(Resource, Clone)]
pub struct AssetProcessor {
    pub(super) server: AssetServer,
    pub(crate) data: Arc<AssetProcessorData>,
}

impl AssetProcessor {
    /// Creates a new [`AssetProcessor`] instance.
    pub fn new(source: &mut AssetSourceBuilders) -> Self {
        let data = Arc::new(AssetProcessorData::new(source.build_sources(true, false)));
        // The asset processor uses its own asset server with its own id space
        let mut sources = source.build_sources(false, false);
        sources.gate_on_processor(data.clone());
        let server = AssetServer::new_with_meta_check(
            sources,
            AssetServerMode::Processed,
            AssetMetaCheck::Always,
            false,
        );
        Self { server, data }
    }

    /// The "internal" [`AssetServer`] used by the [`AssetProcessor`]. This is _separate_ from the asset processor used by
    /// the main App. It has different processor-specific configuration and a different ID space.
    pub fn server(&self) -> &AssetServer {
        &self.server
    }

    async fn set_state(&self, state: ProcessorState) {
        let mut state_guard = self.data.state.write().await;
        let last_state = *state_guard;
        *state_guard = state;
        if last_state != ProcessorState::Finished && state == ProcessorState::Finished {
            self.data.finished_sender.broadcast(()).await.unwrap();
        } else if last_state != ProcessorState::Processing && state == ProcessorState::Processing {
            self.data.initialized_sender.broadcast(()).await.unwrap();
        }
    }

    /// Retrieves the current [`ProcessorState`]
    pub async fn get_state(&self) -> ProcessorState {
        *self.data.state.read().await
    }

    /// Retrieves the [`AssetSource`] for this processor
    #[inline]
    pub fn get_source<'a, 'b>(
        &'a self,
        id: impl Into<AssetSourceId<'b>>,
    ) -> Result<&'a AssetSource, MissingAssetSourceError> {
        self.data.sources.get(id.into())
    }

    #[inline]
    pub fn sources(&self) -> &AssetSources {
        &self.data.sources
    }

    /// Logs an unrecoverable error. On the next run of the processor, all assets will be regenerated. This should only be used as a last resort.
    /// Every call to this should be considered with scrutiny and ideally replaced with something more granular.
    async fn log_unrecoverable(&self) {
        let mut log = self.data.log.write().await;
        let log = log.as_mut().unwrap();
        log.unrecoverable().await.unwrap();
    }

    /// Logs the start of an asset being processed. If this is not followed at some point in the log by a closing [`AssetProcessor::log_end_processing`],
    /// in the next run of the processor the asset processing will be considered "incomplete" and it will be reprocessed.
    async fn log_begin_processing(&self, path: &AssetPath<'_>) {
        let mut log = self.data.log.write().await;
        let log = log.as_mut().unwrap();
        log.begin_processing(path).await.unwrap();
    }

    /// Logs the end of an asset being successfully processed. See [`AssetProcessor::log_begin_processing`].
    async fn log_end_processing(&self, path: &AssetPath<'_>) {
        let mut log = self.data.log.write().await;
        let log = log.as_mut().unwrap();
        log.end_processing(path).await.unwrap();
    }

    /// Starts the processor in a background thread.
    pub fn start(_processor: Res<Self>) {
        #[cfg(any(target_arch = "wasm32", not(feature = "multi-threaded")))]
        error!("Cannot run AssetProcessor in single threaded mode (or WASM) yet.");
        #[cfg(all(not(target_arch = "wasm32"), feature = "multi-threaded"))]
        {
            let processor = _processor.clone();
            std::thread::spawn(move || {
                processor.process_assets();
                bevy_tasks::block_on(processor.listen_for_source_change_events());
            });
        }
    }

    /// Processes all assets. This will:
    /// * For each "processed [`AssetSource`]:
    /// * Scan the [`ProcessorTransactionLog`] and recover from any failures detected
    /// * Scan the processed [`AssetReader`] to build the current view of already processed assets.
    /// * Scan the unprocessed [`AssetReader`] and remove any final processed assets that are invalid or no longer exist.
    /// * For each asset in the unprocessed [`AssetReader`], kick off a new "process job", which will process the asset
    /// (if the latest version of the asset has not been processed).
    #[cfg(all(not(target_arch = "wasm32"), feature = "multi-threaded"))]
    pub fn process_assets(&self) {
        let start_time = std::time::Instant::now();
        debug!("Processing Assets");
        IoTaskPool::get().scope(|scope| {
            scope.spawn(async move {
                self.initialize().await.unwrap();
                for source in self.sources().iter_processed() {
                    self.process_assets_internal(scope, source, PathBuf::from(""))
                        .await
                        .unwrap();
                }
            });
        });
        // This must happen _after_ the scope resolves or it will happen "too early"
        // Don't move this into the async scope above! process_assets is a blocking/sync function this is fine
        bevy_tasks::block_on(self.finish_processing_assets());
        let end_time = std::time::Instant::now();
        debug!("Processing finished in {:?}", end_time - start_time);
    }

    /// Listens for changes to assets in the source [`AssetSource`] and update state accordingly.
    // PERF: parallelize change event processing
    pub async fn listen_for_source_change_events(&self) {
        debug!("Listening for changes to source assets");
        loop {
            let mut started_processing = false;

            for source in self.data.sources.iter_processed() {
                if let Some(receiver) = source.event_receiver() {
                    for event in receiver.try_iter() {
                        if !started_processing {
                            self.set_state(ProcessorState::Processing).await;
                            started_processing = true;
                        }

                        self.handle_asset_source_event(source, event).await;
                    }
                }
            }

            if started_processing {
                self.finish_processing_assets().await;
            }
        }
    }

    async fn handle_asset_source_event(&self, source: &AssetSource, event: AssetSourceEvent) {
        trace!("{event:?}");
        match event {
            AssetSourceEvent::AddedAsset(path)
            | AssetSourceEvent::AddedMeta(path)
            | AssetSourceEvent::ModifiedAsset(path)
            | AssetSourceEvent::ModifiedMeta(path) => {
                self.process_asset(source, path).await;
            }
            AssetSourceEvent::RemovedAsset(path) => {
                self.handle_removed_asset(source, path).await;
            }
            AssetSourceEvent::RemovedMeta(path) => {
                self.handle_removed_meta(source, path).await;
            }
            AssetSourceEvent::AddedFolder(path) => {
                self.handle_added_folder(source, path).await;
            }
            // NOTE: As a heads up for future devs: this event shouldn't be run in parallel with other events that might
            // touch this folder (ex: the folder might be re-created with new assets). Clean up the old state first.
            // Currently this event handler is not parallel, but it could be (and likely should be) in the future.
            AssetSourceEvent::RemovedFolder(path) => {
                self.handle_removed_folder(source, &path).await;
            }
            AssetSourceEvent::RenamedAsset { old, new } => {
                // If there was a rename event, but the path hasn't changed, this asset might need reprocessing.
                // Sometimes this event is returned when an asset is moved "back" into the asset folder
                if old == new {
                    self.process_asset(source, new).await;
                } else {
                    self.handle_renamed_asset(source, old, new).await;
                }
            }
            AssetSourceEvent::RenamedMeta { old, new } => {
                // If there was a rename event, but the path hasn't changed, this asset meta might need reprocessing.
                // Sometimes this event is returned when an asset meta is moved "back" into the asset folder
                if old == new {
                    self.process_asset(source, new).await;
                } else {
                    debug!("Meta renamed from {old:?} to {new:?}");
                    let mut infos = self.data.asset_infos.write().await;
                    // Renaming meta should not assume that an asset has also been renamed. Check both old and new assets to see
                    // if they should be re-imported (and/or have new meta generated)
                    let new_asset_path = AssetPath::from(new).with_source(source.id());
                    let old_asset_path = AssetPath::from(old).with_source(source.id());
                    infos.check_reprocess_queue.push_back(old_asset_path);
                    infos.check_reprocess_queue.push_back(new_asset_path);
                }
            }
            AssetSourceEvent::RenamedFolder { old, new } => {
                // If there was a rename event, but the path hasn't changed, this asset folder might need reprocessing.
                // Sometimes this event is returned when an asset meta is moved "back" into the asset folder
                if old == new {
                    self.handle_added_folder(source, new).await;
                } else {
                    // PERF: this reprocesses everything in the moved folder. this is not necessary in most cases, but
                    // requires some nuance when it comes to path handling.
                    self.handle_removed_folder(source, &old).await;
                    self.handle_added_folder(source, new).await;
                }
            }
            AssetSourceEvent::RemovedUnknown { path, is_meta } => {
                let processed_reader = source.processed_reader().unwrap();
                match processed_reader.is_directory(&path).await {
                    Ok(is_directory) => {
                        if is_directory {
                            self.handle_removed_folder(source, &path).await;
                        } else if is_meta {
                            self.handle_removed_meta(source, path).await;
                        } else {
                            self.handle_removed_asset(source, path).await;
                        }
                    }
                    Err(err) => {
                        match err {
                            AssetReaderError::NotFound(_) => {
                                // if the path is not found, a processed version does not exist
                            }
                            AssetReaderError::Io(err) => {
                                error!(
                                    "Path '{}' was removed, but the destination reader could not determine if it \
                                    was a folder or a file due to the following error: {err}",
                                    AssetPath::from_path(&path).with_source(source.id())
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    async fn handle_added_folder(&self, source: &AssetSource, path: PathBuf) {
        debug!(
            "Folder {} was added. Attempting to re-process",
            AssetPath::from_path(&path).with_source(source.id())
        );
        #[cfg(any(target_arch = "wasm32", not(feature = "multi-threaded")))]
        error!("AddFolder event cannot be handled in single threaded mode (or WASM) yet.");
        #[cfg(all(not(target_arch = "wasm32"), feature = "multi-threaded"))]
        IoTaskPool::get().scope(|scope| {
            scope.spawn(async move {
                self.process_assets_internal(scope, source, path)
                    .await
                    .unwrap();
            });
        });
    }

    /// Responds to a removed meta event by reprocessing the asset at the given path.
    async fn handle_removed_meta(&self, source: &AssetSource, path: PathBuf) {
        // If meta was removed, we might need to regenerate it.
        // Likewise, the user might be manually re-adding the asset.
        // Therefore, we shouldn't automatically delete the asset ... that is a
        // user-initiated action.
        debug!(
            "Meta for asset {:?} was removed. Attempting to re-process",
            AssetPath::from_path(&path).with_source(source.id())
        );
        self.process_asset(source, path).await;
    }

    /// Removes all processed assets stored at the given path (respecting transactionality), then removes the folder itself.
    async fn handle_removed_folder(&self, source: &AssetSource, path: &Path) {
        debug!("Removing folder {:?} because source was removed", path);
        let processed_reader = source.processed_reader().unwrap();
        match processed_reader.read_directory(path).await {
            Ok(mut path_stream) => {
                while let Some(child_path) = path_stream.next().await {
                    self.handle_removed_asset(source, child_path).await;
                }
            }
            Err(err) => match err {
                AssetReaderError::NotFound(_err) => {
                    // The processed folder does not exist. No need to update anything
                }
                AssetReaderError::Io(err) => {
                    self.log_unrecoverable().await;
                    error!(
                        "Unrecoverable Error: Failed to read the processed assets at {path:?} in order to remove assets that no longer exist \
                        in the source directory. Restart the asset processor to fully reprocess assets. Error: {err}"
                    );
                }
            },
        }
        let processed_writer = source.processed_writer().unwrap();
        if let Err(err) = processed_writer.remove_directory(path).await {
            match err {
                AssetWriterError::Io(err) => {
                    // we can ignore NotFound because if the "final" file in a folder was removed
                    // then we automatically clean up this folder
                    if err.kind() != ErrorKind::NotFound {
                        let asset_path = AssetPath::from_path(path).with_source(source.id());
                        error!("Failed to remove destination folder that no longer exists in {asset_path}: {err}");
                    }
                }
            }
        }
    }

    /// Removes the processed version of an asset and associated in-memory metadata. This will block until all existing reads/writes to the
    /// asset have finished, thanks to the `file_transaction_lock`.
    async fn handle_removed_asset(&self, source: &AssetSource, path: PathBuf) {
        let asset_path = AssetPath::from(path).with_source(source.id());
        debug!("Removing processed {asset_path} because source was removed");
        let mut infos = self.data.asset_infos.write().await;
        if let Some(info) = infos.get(&asset_path) {
            // we must wait for uncontested write access to the asset source to ensure existing readers / writers
            // can finish their operations
            let _write_lock = info.file_transaction_lock.write();
            self.remove_processed_asset_and_meta(source, asset_path.path())
                .await;
        }
        infos.remove(&asset_path).await;
    }

    /// Handles a renamed source asset by moving it's processed results to the new location and updating in-memory paths + metadata.
    /// This will cause direct path dependencies to break.
    async fn handle_renamed_asset(&self, source: &AssetSource, old: PathBuf, new: PathBuf) {
        let mut infos = self.data.asset_infos.write().await;
        let old = AssetPath::from(old).with_source(source.id());
        let new = AssetPath::from(new).with_source(source.id());
        let processed_writer = source.processed_writer().unwrap();
        if let Some(info) = infos.get(&old) {
            // we must wait for uncontested write access to the asset source to ensure existing readers / writers
            // can finish their operations
            let _write_lock = info.file_transaction_lock.write();
            processed_writer
                .rename(old.path(), new.path())
                .await
                .unwrap();
            processed_writer
                .rename_meta(old.path(), new.path())
                .await
                .unwrap();
        }
        infos.rename(&old, &new).await;
    }

    async fn finish_processing_assets(&self) {
        self.try_reprocessing_queued().await;
        // clean up metadata in asset server
        self.server.data.infos.write().consume_handle_drop_events();
        self.set_state(ProcessorState::Finished).await;
    }

    #[allow(unused)]
    #[cfg(all(not(target_arch = "wasm32"), feature = "multi-threaded"))]
    fn process_assets_internal<'scope>(
        &'scope self,
        scope: &'scope bevy_tasks::Scope<'scope, '_, ()>,
        source: &'scope AssetSource,
        path: PathBuf,
    ) -> BoxedFuture<'scope, Result<(), AssetReaderError>> {
        Box::pin(async move {
            if source.reader().is_directory(&path).await? {
                let mut path_stream = source.reader().read_directory(&path).await?;
                while let Some(path) = path_stream.next().await {
                    self.process_assets_internal(scope, source, path).await?;
                }
            } else {
                // Files without extensions are skipped
                let processor = self.clone();
                scope.spawn(async move {
                    processor.process_asset(source, path).await;
                });
            }
            Ok(())
        })
    }

    async fn try_reprocessing_queued(&self) {
        loop {
            let mut check_reprocess_queue =
                std::mem::take(&mut self.data.asset_infos.write().await.check_reprocess_queue);
            IoTaskPool::get().scope(|scope| {
                for path in check_reprocess_queue.drain(..) {
                    let processor = self.clone();
                    let source = self.get_source(path.source()).unwrap();
                    scope.spawn(async move {
                        processor.process_asset(source, path.into()).await;
                    });
                }
            });
            let infos = self.data.asset_infos.read().await;
            if infos.check_reprocess_queue.is_empty() {
                break;
            }
        }
    }

    /// Register a new asset processor.
    pub fn register_processor<P: Process>(&self, processor: P) {
        let mut process_plans = self.data.processors.write();
        process_plans.insert(std::any::type_name::<P>(), Arc::new(processor));
    }

    /// Set the default processor for the given `extension`. Make sure `P` is registered with [`AssetProcessor::register_processor`].
    pub fn set_default_processor<P: Process>(&self, extension: &str) {
        let mut default_processors = self.data.default_processors.write();
        default_processors.insert(extension.to_string(), std::any::type_name::<P>());
    }

    /// Returns the default processor for the given `extension`, if it exists.
    pub fn get_default_processor(&self, extension: &str) -> Option<Arc<dyn ErasedProcessor>> {
        let default_processors = self.data.default_processors.read();
        let key = default_processors.get(extension)?;
        self.data.processors.read().get(key).cloned()
    }

    /// Returns the processor with the given `processor_type_name`, if it exists.
    pub fn get_processor(&self, processor_type_name: &str) -> Option<Arc<dyn ErasedProcessor>> {
        let processors = self.data.processors.read();
        processors.get(processor_type_name).cloned()
    }

    /// Populates the initial view of each asset by scanning the unprocessed and processed asset folders.
    /// This info will later be used to determine whether or not to re-process an asset
    ///
    /// This will validate transactions and recover failed transactions when necessary.
    #[allow(unused)]
    async fn initialize(&self) -> Result<(), InitializeError> {
        self.validate_transaction_log_and_recover().await;
        let mut asset_infos = self.data.asset_infos.write().await;

        /// Retrieves asset paths recursively. If `clean_empty_folders_writer` is Some, it will be used to clean up empty
        /// folders when they are discovered.
        fn get_asset_paths<'a>(
            reader: &'a dyn AssetReader,
            clean_empty_folders_writer: Option<&'a dyn AssetWriter>,
            path: PathBuf,
            paths: &'a mut Vec<PathBuf>,
        ) -> BoxedFuture<'a, Result<bool, AssetReaderError>> {
            Box::pin(async move {
                if reader.is_directory(&path).await? {
                    let mut path_stream = reader.read_directory(&path).await?;
                    let mut contains_files = false;
                    while let Some(child_path) = path_stream.next().await {
                        contains_files =
                            get_asset_paths(reader, clean_empty_folders_writer, child_path, paths)
                                .await?
                                && contains_files;
                    }
                    if !contains_files && path.parent().is_some() {
                        if let Some(writer) = clean_empty_folders_writer {
                            // it is ok for this to fail as it is just a cleanup job.
                            let _ = writer.remove_empty_directory(&path).await;
                        }
                    }
                    Ok(contains_files)
                } else {
                    paths.push(path);
                    Ok(true)
                }
            })
        }

        for source in self.sources().iter_processed() {
            let Ok(processed_reader) = source.processed_reader() else {
                continue;
            };
            let Ok(processed_writer) = source.processed_writer() else {
                continue;
            };
            let mut unprocessed_paths = Vec::new();
            get_asset_paths(
                source.reader(),
                None,
                PathBuf::from(""),
                &mut unprocessed_paths,
            )
            .await
            .map_err(InitializeError::FailedToReadSourcePaths)?;

            let mut processed_paths = Vec::new();
            get_asset_paths(
                processed_reader,
                Some(processed_writer),
                PathBuf::from(""),
                &mut processed_paths,
            )
            .await
            .map_err(InitializeError::FailedToReadDestinationPaths)?;

            for path in unprocessed_paths {
                asset_infos.get_or_insert(AssetPath::from(path).with_source(source.id()));
            }

            for path in processed_paths {
                let mut dependencies = Vec::new();
                let asset_path = AssetPath::from(path).with_source(source.id());
                if let Some(info) = asset_infos.get_mut(&asset_path) {
                    match processed_reader.read_meta_bytes(asset_path.path()).await {
                        Ok(meta_bytes) => {
                            match ron::de::from_bytes::<ProcessedInfoMinimal>(&meta_bytes) {
                                Ok(minimal) => {
                                    trace!(
                                        "Populated processed info for asset {asset_path} {:?}",
                                        minimal.processed_info
                                    );

                                    if let Some(processed_info) = &minimal.processed_info {
                                        for process_dependency_info in
                                            &processed_info.process_dependencies
                                        {
                                            dependencies.push(process_dependency_info.path.clone());
                                        }
                                    }
                                    info.processed_info = minimal.processed_info;
                                }
                                Err(err) => {
                                    trace!("Removing processed data for {asset_path} because meta could not be parsed: {err}");
                                    self.remove_processed_asset_and_meta(source, asset_path.path())
                                        .await;
                                }
                            }
                        }
                        Err(err) => {
                            trace!("Removing processed data for {asset_path} because meta failed to load: {err}");
                            self.remove_processed_asset_and_meta(source, asset_path.path())
                                .await;
                        }
                    }
                } else {
                    trace!("Removing processed data for non-existent asset {asset_path}");
                    self.remove_processed_asset_and_meta(source, asset_path.path())
                        .await;
                }

                for dependency in dependencies {
                    asset_infos.add_dependant(&dependency, asset_path.clone());
                }
            }
        }

        self.set_state(ProcessorState::Processing).await;

        Ok(())
    }

    /// Removes the processed version of an asset and its metadata, if it exists. This _is not_ transactional like `remove_processed_asset_transactional`, nor
    /// does it remove existing in-memory metadata.
    async fn remove_processed_asset_and_meta(&self, source: &AssetSource, path: &Path) {
        if let Err(err) = source.processed_writer().unwrap().remove(path).await {
            warn!("Failed to remove non-existent asset {path:?}: {err}");
        }

        if let Err(err) = source.processed_writer().unwrap().remove_meta(path).await {
            warn!("Failed to remove non-existent meta {path:?}: {err}");
        }

        self.clean_empty_processed_ancestor_folders(source, path)
            .await;
    }

    async fn clean_empty_processed_ancestor_folders(&self, source: &AssetSource, path: &Path) {
        // As a safety precaution don't delete absolute paths to avoid deleting folders outside of the destination folder
        if path.is_absolute() {
            error!("Attempted to clean up ancestor folders of an absolute path. This is unsafe so the operation was skipped.");
            return;
        }
        while let Some(parent) = path.parent() {
            if parent == Path::new("") {
                break;
            }
            if source
                .processed_writer()
                .unwrap()
                .remove_empty_directory(parent)
                .await
                .is_err()
            {
                // if we fail to delete a folder, stop walking up the tree
                break;
            }
        }
    }

    /// Processes the asset (if it has not already been processed or the asset source has changed).
    /// If the asset has "process dependencies" (relies on the values of other assets), it will asynchronously await until
    /// the dependencies have been processed (See [`ProcessorGatedReader`], which is used in the [`AssetProcessor`]'s [`AssetServer`]
    /// to block reads until the asset is processed).
    ///
    /// [`LoadContext`]: crate::loader::LoadContext
    /// [`ProcessorGatedReader`]: crate::io::processor_gated::ProcessorGatedReader
    async fn process_asset(&self, source: &AssetSource, path: PathBuf) {
        let asset_path = AssetPath::from(path).with_source(source.id());
        let result = self.process_asset_internal(source, &asset_path).await;
        let mut infos = self.data.asset_infos.write().await;
        infos.finish_processing(asset_path, result).await;
    }

    async fn process_asset_internal(
        &self,
        source: &AssetSource,
        asset_path: &AssetPath<'static>,
    ) -> Result<ProcessResult, ProcessError> {
        // TODO: The extension check was removed now that AssetPath is the input. is that ok?
        // TODO: check if already processing to protect against duplicate hot-reload events
        debug!("Processing {:?}", asset_path);
        let server = &self.server;
        let path = asset_path.path();
        let reader = source.reader();

        let reader_err = |err| ProcessError::AssetReaderError {
            path: asset_path.clone(),
            err,
        };
        let writer_err = |err| ProcessError::AssetWriterError {
            path: asset_path.clone(),
            err,
        };

        // Note: we get the asset source reader first because we don't want to create meta files for assets that don't have source files
        let mut byte_reader = reader.read(path).await.map_err(reader_err)?;

        let (mut source_meta, meta_bytes, processor) = match reader.read_meta_bytes(path).await {
            Ok(meta_bytes) => {
                let minimal: AssetMetaMinimal = ron::de::from_bytes(&meta_bytes).map_err(|e| {
                    ProcessError::DeserializeMetaError(DeserializeMetaError::DeserializeMinimal(e))
                })?;
                let (meta, processor) = match minimal.asset {
                    AssetActionMinimal::Load { loader } => {
                        let loader = server.get_asset_loader_with_type_name(&loader).await?;
                        let meta = loader.deserialize_meta(&meta_bytes)?;
                        (meta, None)
                    }
                    AssetActionMinimal::Process { processor } => {
                        let processor = self
                            .get_processor(&processor)
                            .ok_or_else(|| ProcessError::MissingProcessor(processor))?;
                        let meta = processor.deserialize_meta(&meta_bytes)?;
                        (meta, Some(processor))
                    }
                    AssetActionMinimal::Ignore => {
                        let meta: Box<dyn AssetMetaDyn> =
                            Box::new(AssetMeta::<(), ()>::deserialize(&meta_bytes)?);
                        (meta, None)
                    }
                };
                (meta, meta_bytes, processor)
            }
            Err(AssetReaderError::NotFound(_path)) => {
                let (meta, processor) = if let Some(processor) = asset_path
                    .get_full_extension()
                    .and_then(|ext| self.get_default_processor(&ext))
                {
                    let meta = processor.default_meta();
                    (meta, Some(processor))
                } else {
                    match server.get_path_asset_loader(asset_path.clone()).await {
                        Ok(loader) => (loader.default_meta(), None),
                        Err(MissingAssetLoaderForExtensionError { .. }) => {
                            let meta: Box<dyn AssetMetaDyn> =
                                Box::new(AssetMeta::<(), ()>::new(AssetAction::Ignore));
                            (meta, None)
                        }
                    }
                };
                let meta_bytes = meta.serialize();
                // write meta to source location if it doesn't already exist
                source
                    .writer()?
                    .write_meta_bytes(path, &meta_bytes)
                    .await
                    .map_err(writer_err)?;
                (meta, meta_bytes, processor)
            }
            Err(err) => {
                return Err(ProcessError::ReadAssetMetaError {
                    path: asset_path.clone(),
                    err,
                })
            }
        };

        let processed_writer = source.processed_writer()?;

        let mut asset_bytes = Vec::new();
        byte_reader
            .read_to_end(&mut asset_bytes)
            .await
            .map_err(|e| ProcessError::AssetReaderError {
                path: asset_path.clone(),
                err: AssetReaderError::Io(e),
            })?;

        // PERF: in theory these hashes could be streamed if we want to avoid allocating the whole asset.
        // The downside is that reading assets would need to happen twice (once for the hash and once for the asset loader)
        // Hard to say which is worse
        let new_hash = get_asset_hash(&meta_bytes, &asset_bytes);
        let mut new_processed_info = ProcessedInfo {
            hash: new_hash,
            full_hash: new_hash,
            process_dependencies: Vec::new(),
        };

        {
            let infos = self.data.asset_infos.read().await;
            if let Some(current_processed_info) = infos
                .get(asset_path)
                .and_then(|i| i.processed_info.as_ref())
            {
                if current_processed_info.hash == new_hash {
                    let mut dependency_changed = false;
                    for current_dep_info in &current_processed_info.process_dependencies {
                        let live_hash = infos
                            .get(&current_dep_info.path)
                            .and_then(|i| i.processed_info.as_ref())
                            .map(|i| i.full_hash);
                        if live_hash != Some(current_dep_info.full_hash) {
                            dependency_changed = true;
                            break;
                        }
                    }
                    if !dependency_changed {
                        return Ok(ProcessResult::SkippedNotChanged);
                    }
                }
            }
        }
        // Note: this lock must remain alive until all processed asset asset and meta writes have finished (or failed)
        // See ProcessedAssetInfo::file_transaction_lock docs for more info
        let _transaction_lock = {
            let mut infos = self.data.asset_infos.write().await;
            let info = infos.get_or_insert(asset_path.clone());
            info.file_transaction_lock.write_arc().await
        };

        // NOTE: if processing the asset fails this will produce an "unfinished" log entry, forcing a rebuild on next run.
        // Directly writing to the asset destination in the processor necessitates this behavior
        // TODO: this class of failure can be recovered via re-processing + smarter log validation that allows for duplicate transactions in the event of failures
        self.log_begin_processing(asset_path).await;
        if let Some(processor) = processor {
            let mut writer = processed_writer.write(path).await.map_err(writer_err)?;
            let mut processed_meta = {
                let mut context =
                    ProcessContext::new(self, asset_path, &asset_bytes, &mut new_processed_info);
                processor
                    .process(&mut context, source_meta, &mut *writer)
                    .await?
            };

            writer
                .flush()
                .await
                .map_err(|e| ProcessError::AssetWriterError {
                    path: asset_path.clone(),
                    err: AssetWriterError::Io(e),
                })?;

            let full_hash = get_full_asset_hash(
                new_hash,
                new_processed_info
                    .process_dependencies
                    .iter()
                    .map(|i| i.full_hash),
            );
            new_processed_info.full_hash = full_hash;
            *processed_meta.processed_info_mut() = Some(new_processed_info.clone());
            let meta_bytes = processed_meta.serialize();
            processed_writer
                .write_meta_bytes(path, &meta_bytes)
                .await
                .map_err(writer_err)?;
        } else {
            processed_writer
                .write_bytes(path, &asset_bytes)
                .await
                .map_err(writer_err)?;
            *source_meta.processed_info_mut() = Some(new_processed_info.clone());
            let meta_bytes = source_meta.serialize();
            processed_writer
                .write_meta_bytes(path, &meta_bytes)
                .await
                .map_err(writer_err)?;
        }
        self.log_end_processing(asset_path).await;

        Ok(ProcessResult::Processed(new_processed_info))
    }

    async fn validate_transaction_log_and_recover(&self) {
        if let Err(err) = ProcessorTransactionLog::validate().await {
            let state_is_valid = match err {
                ValidateLogError::ReadLogError(err) => {
                    error!("Failed to read processor log file. Processed assets cannot be validated so they must be re-generated {err}");
                    false
                }
                ValidateLogError::UnrecoverableError => {
                    error!("Encountered an unrecoverable error in the last run. Processed assets cannot be validated so they must be re-generated");
                    false
                }
                ValidateLogError::EntryErrors(entry_errors) => {
                    let mut state_is_valid = true;
                    for entry_error in entry_errors {
                        match entry_error {
                            LogEntryError::DuplicateTransaction(_)
                            | LogEntryError::EndedMissingTransaction(_) => {
                                error!("{}", entry_error);
                                state_is_valid = false;
                                break;
                            }
                            LogEntryError::UnfinishedTransaction(path) => {
                                debug!("Asset {path:?} did not finish processing. Clearing state for that asset");
                                let mut unrecoverable_err = |message: &dyn std::fmt::Display| {
                                    error!("Failed to remove asset {path:?}: {message}");
                                    state_is_valid = false;
                                };
                                let Ok(source) = self.get_source(path.source()) else {
                                    (unrecoverable_err)(&"AssetSource does not exist");
                                    continue;
                                };
                                let Ok(processed_writer) = source.processed_writer() else {
                                    (unrecoverable_err)(&"AssetSource does not have a processed AssetWriter registered");
                                    continue;
                                };

                                if let Err(err) = processed_writer.remove(path.path()).await {
                                    match err {
                                        AssetWriterError::Io(err) => {
                                            // any error but NotFound means we could be in a bad state
                                            if err.kind() != ErrorKind::NotFound {
                                                (unrecoverable_err)(&err);
                                            }
                                        }
                                    }
                                }
                                if let Err(err) = processed_writer.remove_meta(path.path()).await {
                                    match err {
                                        AssetWriterError::Io(err) => {
                                            // any error but NotFound means we could be in a bad state
                                            if err.kind() != ErrorKind::NotFound {
                                                (unrecoverable_err)(&err);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    state_is_valid
                }
            };

            if !state_is_valid {
                error!("Processed asset transaction log state was invalid and unrecoverable for some reason (see previous logs). Removing processed assets and starting fresh.");
                for source in self.sources().iter_processed() {
                    let Ok(processed_writer) = source.processed_writer() else {
                        continue;
                    };
                    if let Err(err) = processed_writer
                        .remove_assets_in_directory(Path::new(""))
                        .await
                    {
                        panic!("Processed assets were in a bad state. To correct this, the asset processor attempted to remove all processed assets and start from scratch. This failed. There is no way to continue. Try restarting, or deleting imported asset folder manually. {err}");
                    }
                }
            }
        }
        let mut log = self.data.log.write().await;
        *log = match ProcessorTransactionLog::new().await {
            Ok(log) => Some(log),
            Err(err) => panic!("Failed to initialize asset processor log. This cannot be recovered. Try restarting. If that doesn't work, try deleting processed asset folder. {}", err),
        };
    }
}
