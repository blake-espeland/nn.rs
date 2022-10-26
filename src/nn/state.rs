use crate::util::dtypes::*;

use super::network::Network;

/*
Maintains the state of the network during training and inference.
*/
pub struct NetworkState {
    pub truth: Array2F, // Data loaded from dataloader
    pub input: Array4F, // input to current layer
    pub delta: Array2F, // dl/dL -> how much the loss for output is affected by layer

    pub workspace: Array4F, // Not sure, probably has to do with Cuda

    pub train: bool,  // Are we training?
    pub index: Int,   // What layer?
    pub net: Network, // Container for network layers and hyperparameters
}

impl Default for NetworkState {
    fn default() -> Self {
        NetworkState {
            truth: Array2F::zeros([0, 0]),
            input: Array4F::zeros([0, 0, 0, 0]),
            delta: Array2F::zeros([0, 0]),
            workspace: Array4F::zeros([0, 0, 0, 0]),
            train: true,
            index: 0,
            net: Network::default(),
        }
    }
}
