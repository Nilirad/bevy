pub use super::*;

use crate::{io::AssetSources, AssetPath};

use parking_lot::RwLock;
use std::sync::Arc;

pub struct AssetProcessorData {
    pub(crate) asset_infos: async_lock::RwLock<ProcessorAssetInfos>,
    pub(super) log: async_lock::RwLock<Option<ProcessorTransactionLog>>,
    pub(super) processors: RwLock<HashMap<&'static str, Arc<dyn ErasedProcessor>>>,
    /// Default processors for file extensions
    pub(super) default_processors: RwLock<HashMap<String, &'static str>>,
    pub(super) state: async_lock::RwLock<ProcessorState>,
    pub(super) sources: AssetSources,
    pub(super) initialized_sender: async_broadcast::Sender<()>,
    initialized_receiver: async_broadcast::Receiver<()>,
    pub(super) finished_sender: async_broadcast::Sender<()>,
    finished_receiver: async_broadcast::Receiver<()>,
}

impl AssetProcessorData {
    pub fn new(source: AssetSources) -> Self {
        let (mut finished_sender, finished_receiver) = async_broadcast::broadcast(1);
        let (mut initialized_sender, initialized_receiver) = async_broadcast::broadcast(1);
        // allow overflow on these "one slot" channels to allow receivers to retrieve the "latest" state, and to allow senders to
        // not block if there was older state present.
        finished_sender.set_overflow(true);
        initialized_sender.set_overflow(true);

        AssetProcessorData {
            sources: source,
            finished_sender,
            finished_receiver,
            initialized_sender,
            initialized_receiver,
            state: async_lock::RwLock::new(ProcessorState::Initializing),
            log: Default::default(),
            processors: Default::default(),
            asset_infos: Default::default(),
            default_processors: Default::default(),
        }
    }

    /// Returns a future that will not finish until the path has been processed.
    pub async fn wait_until_processed(&self, path: AssetPath<'static>) -> ProcessStatus {
        self.wait_until_initialized().await;
        let mut receiver = {
            let infos = self.asset_infos.write().await;
            let info = infos.get(&path);
            match info {
                Some(info) => match info.status {
                    Some(result) => return result,
                    // This receiver must be created prior to losing the read lock to ensure this is transactional
                    None => info.status_receiver.clone(),
                },
                None => return ProcessStatus::NonExistent,
            }
        };
        receiver.recv().await.unwrap()
    }

    /// Returns a future that will not finish until the processor has been initialized.
    pub async fn wait_until_initialized(&self) {
        let receiver = {
            let state = self.state.read().await;
            match *state {
                ProcessorState::Initializing => {
                    // This receiver must be created prior to losing the read lock to ensure this is transactional
                    Some(self.initialized_receiver.clone())
                }
                _ => None,
            }
        };

        if let Some(mut receiver) = receiver {
            receiver.recv().await.unwrap();
        }
    }

    /// Returns a future that will not finish until processing has finished.
    pub async fn wait_until_finished(&self) {
        let receiver = {
            let state = self.state.read().await;
            match *state {
                ProcessorState::Initializing | ProcessorState::Processing => {
                    // This receiver must be created prior to losing the read lock to ensure this is transactional
                    Some(self.finished_receiver.clone())
                }
                ProcessorState::Finished => None,
            }
        };

        if let Some(mut receiver) = receiver {
            receiver.recv().await.unwrap();
        }
    }
}
