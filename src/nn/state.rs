use crate::util::dtypes::*;
use ndarray::Array;

use super::network::Network;

/*
Maintains the state of the network during training and inference.
*/
pub struct NetworkState {
    truth: Array<Float, usize>, // Data loaded from dataloader
    input: Array<Float, usize>, // input to current layer
    delta: Array<Float, usize>, // for backprop

    workspace: Array<Float, usize>, // Not sure

    train: bool,  // Are we training?
    index: Int,   // What layer?
    net: Network, // Container for network layers and hyperparameters
}
