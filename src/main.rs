mod blas;
mod nn;
mod util;

use ndarray::{Array, IxDyn};
use nn::state::{NetworkState};
use util::dtypes::*;

fn main() {
    let mut net = util::parser::parse_cfg("cfg/model.json");
    let dataset = util::data::Dataset::new(&net.data_path).unwrap();
    let loader = util::data::DataLoader::new(dataset, net.batch_size);

    let mut state = NetworkState::default();

    state.net = net;
    state.train = true; // Needs to be changed when adding inference option
}
