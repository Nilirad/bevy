mod asset_processor;
mod asset_processor_data;
mod log;
mod process;

pub use asset_processor::*;
pub use asset_processor_data::*;
pub use log::*;
pub use process::*;

use crate::{
    io::AssetReaderError,
    meta::{AssetHash, ProcessedInfo},
    AssetLoadError, AssetPath,
};
use bevy_ecs::prelude::*;
use bevy_log::{debug, error, trace};

use bevy_utils::{HashMap, HashSet};

use std::{collections::VecDeque, sync::Arc};
use thiserror::Error;

/// The (successful) result of processing an asset
#[derive(Debug, Clone)]
pub enum ProcessResult {
    Processed(ProcessedInfo),
    SkippedNotChanged,
}

/// The final status of processing an asset
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum ProcessStatus {
    Processed,
    Failed,
    NonExistent,
}

// NOTE: if you add new fields to this struct, make sure they are propagated (when relevant) in ProcessorAssetInfos::rename
#[derive(Debug)]
pub(crate) struct ProcessorAssetInfo {
    processed_info: Option<ProcessedInfo>,
    /// Paths of assets that depend on this asset when they are being processed.
    dependants: HashSet<AssetPath<'static>>,
    status: Option<ProcessStatus>,
    /// A lock that controls read/write access to processed asset files. The lock is shared for both the asset bytes and the meta bytes.
    /// _This lock must be locked whenever a read or write to processed assets occurs_
    /// There are scenarios where processed assets (and their metadata) are being read and written in multiple places at once:
    /// * when the processor is running in parallel with an app
    /// * when processing assets in parallel, the processor might read an asset's process_dependencies when processing new versions of those dependencies
    ///     * this second scenario almost certainly isn't possible with the current implementation, but its worth protecting against
    /// This lock defends against those scenarios by ensuring readers don't read while processed files are being written. And it ensures
    /// Because this lock is shared across meta and asset bytes, readers can ensure they don't read "old" versions of metadata with "new" asset data.
    pub(crate) file_transaction_lock: Arc<async_lock::RwLock<()>>,
    status_sender: async_broadcast::Sender<ProcessStatus>,
    status_receiver: async_broadcast::Receiver<ProcessStatus>,
}

impl Default for ProcessorAssetInfo {
    fn default() -> Self {
        let (mut status_sender, status_receiver) = async_broadcast::broadcast(1);
        // allow overflow on these "one slot" channels to allow receivers to retrieve the "latest" state, and to allow senders to
        // not block if there was older state present.
        status_sender.set_overflow(true);
        Self {
            processed_info: Default::default(),
            dependants: Default::default(),
            file_transaction_lock: Default::default(),
            status: None,
            status_sender,
            status_receiver,
        }
    }
}

impl ProcessorAssetInfo {
    async fn update_status(&mut self, status: ProcessStatus) {
        if self.status != Some(status) {
            self.status = Some(status);
            self.status_sender.broadcast(status).await.unwrap();
        }
    }
}

/// The "current" in memory view of the asset space. This is "eventually consistent". It does not directly
/// represent the state of assets in storage, but rather a valid historical view that will gradually become more
/// consistent as events are processed.
#[derive(Default, Debug)]
pub struct ProcessorAssetInfos {
    /// The "current" in memory view of the asset space. During processing, if path does not exist in this, it should
    /// be considered non-existent.
    /// NOTE: YOU MUST USE `Self::get_or_insert` or `Self::insert` TO ADD ITEMS TO THIS COLLECTION TO ENSURE
    /// non_existent_dependants DATA IS CONSUMED
    infos: HashMap<AssetPath<'static>, ProcessorAssetInfo>,
    /// Dependants for assets that don't exist. This exists to track "dangling" asset references due to deleted / missing files.
    /// If the dependant asset is added, it can "resolve" these dependencies and re-compute those assets.
    /// Therefore this _must_ always be consistent with the `infos` data. If a new asset is added to `infos`, it should
    /// check this maps for dependencies and add them. If an asset is removed, it should update the dependants here.
    non_existent_dependants: HashMap<AssetPath<'static>, HashSet<AssetPath<'static>>>,
    check_reprocess_queue: VecDeque<AssetPath<'static>>,
}

impl ProcessorAssetInfos {
    fn get_or_insert(&mut self, asset_path: AssetPath<'static>) -> &mut ProcessorAssetInfo {
        self.infos.entry(asset_path.clone()).or_insert_with(|| {
            let mut info = ProcessorAssetInfo::default();
            // track existing dependants by resolving existing "hanging" dependants.
            if let Some(dependants) = self.non_existent_dependants.remove(&asset_path) {
                info.dependants = dependants;
            }
            info
        })
    }

    pub(crate) fn get(&self, asset_path: &AssetPath<'static>) -> Option<&ProcessorAssetInfo> {
        self.infos.get(asset_path)
    }

    fn get_mut(&mut self, asset_path: &AssetPath<'static>) -> Option<&mut ProcessorAssetInfo> {
        self.infos.get_mut(asset_path)
    }

    fn add_dependant(&mut self, asset_path: &AssetPath<'static>, dependant: AssetPath<'static>) {
        if let Some(info) = self.get_mut(asset_path) {
            info.dependants.insert(dependant);
        } else {
            let dependants = self
                .non_existent_dependants
                .entry(asset_path.clone())
                .or_default();
            dependants.insert(dependant);
        }
    }

    /// Finalize processing the asset, which will incorporate the result of the processed asset into the in-memory view the processed assets.
    async fn finish_processing(
        &mut self,
        asset_path: AssetPath<'static>,
        result: Result<ProcessResult, ProcessError>,
    ) {
        match result {
            Ok(ProcessResult::Processed(processed_info)) => {
                debug!("Finished processing \"{:?}\"", asset_path);
                // clean up old dependants
                let old_processed_info = self
                    .infos
                    .get_mut(&asset_path)
                    .and_then(|i| i.processed_info.take());
                if let Some(old_processed_info) = old_processed_info {
                    self.clear_dependencies(&asset_path, old_processed_info);
                }

                // populate new dependants
                for process_dependency_info in &processed_info.process_dependencies {
                    self.add_dependant(&process_dependency_info.path, asset_path.to_owned());
                }
                let info = self.get_or_insert(asset_path);
                info.processed_info = Some(processed_info);
                info.update_status(ProcessStatus::Processed).await;
                let dependants = info.dependants.iter().cloned().collect::<Vec<_>>();
                for path in dependants {
                    self.check_reprocess_queue.push_back(path);
                }
            }
            Ok(ProcessResult::SkippedNotChanged) => {
                debug!("Skipping processing (unchanged) \"{:?}\"", asset_path);
                let info = self.get_mut(&asset_path).expect("info should exist");
                // NOTE: skipping an asset on a given pass doesn't mean it won't change in the future as a result
                // of a dependency being re-processed. This means apps might receive an "old" (but valid) asset first.
                // This is in the interest of fast startup times that don't block for all assets being checked + reprocessed
                // Therefore this relies on hot-reloading in the app to pickup the "latest" version of the asset
                // If "block until latest state is reflected" is required, we can easily add a less granular
                // "block until first pass finished" mode
                info.update_status(ProcessStatus::Processed).await;
            }
            Err(ProcessError::ExtensionRequired) => {
                // Skip assets without extensions
            }
            Err(ProcessError::MissingAssetLoaderForExtension(_)) => {
                trace!("No loader found for {asset_path}");
            }
            Err(ProcessError::AssetReaderError {
                err: AssetReaderError::NotFound(_),
                ..
            }) => {
                // if there is no asset source, no processing can be done
                trace!("No need to process asset {asset_path} because it does not exist");
            }
            Err(err) => {
                error!("Failed to process asset {asset_path}: {err}");
                // if this failed because a dependency could not be loaded, make sure it is reprocessed if that dependency is reprocessed
                if let ProcessError::AssetLoadError(AssetLoadError::AssetLoaderError {
                    path: dependency,
                    ..
                }) = err
                {
                    let info = self.get_mut(&asset_path).expect("info should exist");
                    info.processed_info = Some(ProcessedInfo {
                        hash: AssetHash::default(),
                        full_hash: AssetHash::default(),
                        process_dependencies: vec![],
                    });
                    self.add_dependant(&dependency, asset_path.to_owned());
                }

                let info = self.get_mut(&asset_path).expect("info should exist");
                info.update_status(ProcessStatus::Failed).await;
            }
        }
    }

    /// Remove the info for the given path. This should only happen if an asset's source is removed / non-existent
    async fn remove(&mut self, asset_path: &AssetPath<'static>) {
        let info = self.infos.remove(asset_path);
        if let Some(info) = info {
            if let Some(processed_info) = info.processed_info {
                self.clear_dependencies(asset_path, processed_info);
            }
            // Tell all listeners this asset does not exist
            info.status_sender
                .broadcast(ProcessStatus::NonExistent)
                .await
                .unwrap();
            if !info.dependants.is_empty() {
                error!(
                    "The asset at {asset_path} was removed, but it had assets that depend on it to be processed. Consider updating the path in the following assets: {:?}",
                    info.dependants
                );
                self.non_existent_dependants
                    .insert(asset_path.clone(), info.dependants);
            }
        }
    }

    /// Remove the info for the given path. This should only happen if an asset's source is removed / non-existent
    async fn rename(&mut self, old: &AssetPath<'static>, new: &AssetPath<'static>) {
        let info = self.infos.remove(old);
        if let Some(mut info) = info {
            if !info.dependants.is_empty() {
                // TODO: We can't currently ensure "moved" folders with relative paths aren't broken because AssetPath
                // doesn't distinguish between absolute and relative paths. We have "erased" relativeness. In the short term,
                // we could do "remove everything in a folder and re-add", but that requires full rebuilds / destroying the cache.
                // If processors / loaders could enumerate dependencies, we could check if the new deps line up with a rename.
                // If deps encoded "relativeness" as part of loading, that would also work (this seems like the right call).
                // TODO: it would be nice to log an error here for dependants that aren't also being moved + fixed.
                // (see the remove impl).
                error!(
                    "The asset at {old} was removed, but it had assets that depend on it to be processed. Consider updating the path in the following assets: {:?}",
                    info.dependants
                );
                self.non_existent_dependants
                    .insert(old.clone(), std::mem::take(&mut info.dependants));
            }
            if let Some(processed_info) = &info.processed_info {
                // Update "dependant" lists for this asset's "process dependencies" to use new path.
                for dep in &processed_info.process_dependencies {
                    if let Some(info) = self.infos.get_mut(&dep.path) {
                        info.dependants.remove(old);
                        info.dependants.insert(new.clone());
                    } else if let Some(dependants) = self.non_existent_dependants.get_mut(&dep.path)
                    {
                        dependants.remove(old);
                        dependants.insert(new.clone());
                    }
                }
            }
            // Tell all listeners this asset no longer exists
            info.status_sender
                .broadcast(ProcessStatus::NonExistent)
                .await
                .unwrap();
            let dependants: Vec<AssetPath<'static>> = {
                let new_info = self.get_or_insert(new.clone());
                new_info.processed_info = info.processed_info;
                new_info.status = info.status;
                // Ensure things waiting on the new path are informed of the status of this asset
                if let Some(status) = new_info.status {
                    new_info.status_sender.broadcast(status).await.unwrap();
                }
                new_info.dependants.iter().cloned().collect()
            };
            // Queue the asset for a reprocess check, in case it needs new meta.
            self.check_reprocess_queue.push_back(new.clone());
            for dependant in dependants {
                // Queue dependants for reprocessing because they might have been waiting for this asset.
                self.check_reprocess_queue.push_back(dependant);
            }
        }
    }

    fn clear_dependencies(&mut self, asset_path: &AssetPath<'static>, removed_info: ProcessedInfo) {
        for old_load_dep in removed_info.process_dependencies {
            if let Some(info) = self.infos.get_mut(&old_load_dep.path) {
                info.dependants.remove(asset_path);
            } else if let Some(dependants) =
                self.non_existent_dependants.get_mut(&old_load_dep.path)
            {
                dependants.remove(asset_path);
            }
        }
    }
}

/// The current state of the [`AssetProcessor`].
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ProcessorState {
    /// The processor is still initializing, which involves scanning the current asset folders,
    /// constructing an in-memory view of the asset space, recovering from previous errors / crashes,
    /// and cleaning up old / unused assets.
    Initializing,
    /// The processor is currently processing assets.
    Processing,
    /// The processor has finished processing all valid assets and reporting invalid assets.
    Finished,
}

/// An error that occurs when initializing the [`AssetProcessor`].
#[derive(Error, Debug)]
pub enum InitializeError {
    #[error(transparent)]
    FailedToReadSourcePaths(AssetReaderError),
    #[error(transparent)]
    FailedToReadDestinationPaths(AssetReaderError),
    #[error("Failed to validate asset log: {0}")]
    ValidateLogError(ValidateLogError),
}
