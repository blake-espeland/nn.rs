use ndarray::{IxDyn};

use crate::util::dtypes::*;
use super::Activation;

use super::{
    activation::{get_act_fn, get_gradient_fn},
    layer::Layer,
    state::NetworkState,
};

trait ConvolutionalLayer {
    fn new_conv(
        batch_no: usize,
        steps: usize,
        kernel: usize,
        act: Activation,
        i_shape: Conv2dShape,
        o_shape: Conv2dShape,
        pad: &Two<usize>,
        stride: &Two<usize>,
    ) -> Self;
    fn forward_conv(&mut self, net: &NetworkState) -> ();
    fn backward_conv(&mut self, net: &NetworkState) -> ();
    fn update_conv(&mut self, net: &NetworkState) -> ();
}

impl ConvolutionalLayer for Layer {
    fn new_conv(
        batch_no: usize,
        steps: usize,
        kernel: usize,
        act: Activation,
        i_shape: Conv2dShape,
        o_shape: Conv2dShape,
        pad: &Two<usize>,
        stride: &Two<usize>,
    ) -> Layer {

        let n = o_shape.c;
        let c = i_shape.c;

        let w_shape = IxDyn(&[c, n, kernel, kernel]);
        let b_shape = IxDyn(&[n]);
        let x_shape = IxDyn(&[1]);

        Layer {
            cur_batch: batch_no,

            act: get_act_fn(&act),
            act_b: get_gradient_fn(&act),
            layer_delta: FloatArr::zeros(x_shape.clone()),

            outputs: FloatArr::zeros(IxDyn(&o_shape.to_arr())),
            inputs: FloatArr::zeros(IxDyn(&i_shape.to_arr())),

            input_layers: Vec::<usize>::new(),

            weights: FloatArr::zeros(w_shape),
            biases: FloatArr::zeros(b_shape),
            loss: FloatArr::zeros(x_shape.clone()),

            t: steps,
            b: i_shape.b,
            h: i_shape.h,
            w: i_shape.w,
            c: c,
            n: n,

            kernel_size: kernel,
            stride: stride.clone(),
            pad: pad.clone(),
        }
    }
    fn forward_conv(&mut self, net: &NetworkState) -> () {}
    fn backward_conv(&mut self, net: &NetworkState) -> () {}
    fn update_conv(&mut self, net: &NetworkState) -> () {}
}
