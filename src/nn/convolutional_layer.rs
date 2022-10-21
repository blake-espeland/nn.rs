use ndarray::IxDyn;

use super::{get_act_str, Activation};
use crate::util::dtypes::*;

use super::{
    activation::{get_act_fn, get_gradient_fn},
    layer::Layer,
    state::NetworkState,
};

pub trait ConvolutionalLayer {
    fn new_conv(
        batch_no: usize,
        steps: usize,
        kernel: usize,
        act_fn: Activation,
        i_shape: DataShape,
        o_shape: DataShape,
        pad: &Two<usize>,
        stride: &Two<usize>,
    ) -> Self;
    fn forward_conv(&mut self, net: &NetworkState) -> ();
    fn backward_conv(&mut self, net: &NetworkState) -> ();
    fn update_conv(&mut self, net: &NetworkState) -> ();
    fn print_conv(&self) -> ();
}

impl ConvolutionalLayer for Layer {
    fn print_conv(&self) {
        let o = self.outputs.shape();
        println!(
            "Convolutional ({})\n\
            \t {} filters \n \
            \t {}x{} kernel \n \
            \t {}x{} stride \n \
            \t {}x{} pad \n \
            \t {}x{}x{} -> {}x{}x{} \n \
            ",
            get_act_str(self.act),
            self.n,
            self.kernel_size,
            self.kernel_size,
            self.stride.x,
            self.stride.y,
            self.pad.x,
            self.pad.y,
            self.h,
            self.w,
            self.c,
            o[1],
            o[2],
            o[3]
        )
    }

    fn new_conv(
        batch_no: usize,
        steps: usize,
        kernel: usize,
        act_fn: Activation,
        i_shape: DataShape,
        o_shape: DataShape,
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

            act: act_fn.clone(),
            act_fn: get_act_fn(&act_fn),
            act_grad: get_gradient_fn(&act_fn),
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
