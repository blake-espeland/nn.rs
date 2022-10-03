use super::activation::*;
use super::node::*;

pub struct Connection{
    start: &Node,
    end: &Node,
    weight: Nval,
    activation: Activation
}

impl Connection{}

impl Default for Connection{
    fn default(){
        Connection{
            start: Node::default(),
            end: Node::default(),
            weight: 0.0,
            activation: RELU
        }
    }
}