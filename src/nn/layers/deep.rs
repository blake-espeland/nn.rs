use crate::nn::node::{Node};
use crate::nn::activation::Activation as Activation;
use super::traits as traits;

use rand::prelude::*;

struct DeepLayer{
    input: bool,
    output: bool,
    forward: fn(x: c_float) -> c_float,
    backward: fn(x: c_float) -> c_float,
    nodes: Vec<Nodes>,
}

impl traits::LayerTrait for DeepLayer{
    fn forward_pass(x: c_float) -> c_float {

    }

    fn backward_pass(x: c_float) -> c_float{

    }

    fn init_random(&mut self){
        for node in self.nodes{
            node.update(rand::thread_rng().rng());
        }
    }
}

impl Default for DeepLayer{
    fn default() -> DeepLayer{
        DeepLayer {
            input: false,
            output: false,
            forward: id_f,
            backward: id_b,
            nodes: Vec::default()
        }
    }
}