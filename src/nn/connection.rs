use super::activation::*;
use super::node::*;

pub struct Connection{
    weight: Nval,
    activation: Activation
}

impl Connection{}

impl Default for Connection{
    fn default(){
        Connection{
            weight: 0.0,
            activation: RELU
        }
    }
}