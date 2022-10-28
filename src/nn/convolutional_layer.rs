use std::cmp::max;

use ndarray::IxDyn;

use super::{activation::activate_array, get_act_str, Activation, CostType, LayerType};
use crate::{
    blas::{convolutional_out_height, convolutional_out_width},
    util::dtypes::*,
};

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
        i_shape: Array4Shape,
        o_shape: Array4Shape,
        pad: &Two<usize>,
        stride: &Two<usize>,
    ) -> Self;
    fn forward_conv(&mut self, state: &mut NetworkState) -> ();
    fn backward_conv(&mut self, state: &mut NetworkState) -> ();
    fn update_conv(&mut self, state: &mut NetworkState) -> ();
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
        i_shape: Array4Shape,
        o_shape: Array4Shape,
        pad: &Two<usize>,
        stride: &Two<usize>,
    ) -> Layer {
        let n = o_shape.c;
        let c = i_shape.c;

        let w_shape = [kernel, kernel, c, n];
        let bias_shape = [n];
        let singleton_shape = [1];

        Layer {
            layer_type: LayerType::Conv,
            cost_type: CostType::L1,

            cur_batch: batch_no,

            act: act_fn.clone(),
            act_fn: get_act_fn(&act_fn),
            act_grad: get_gradient_fn(&act_fn),
            delta: Array1F::zeros(singleton_shape),

            outputs: Array4F::zeros(o_shape.to_arr()),
            inputs: Array4F::zeros(i_shape.to_arr()),

            input_layers: Vec::<usize>::new(),

            weights: Array4F::zeros(w_shape),
            biases: Array1F::zeros(bias_shape),
            loss: Array1F::zeros(singleton_shape),

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

    fn forward_conv(&mut self, state: &mut NetworkState) -> () {
        self.inputs.clone_from(&state.input);

        // kernel -> [k, k, c, n]
        // input -> [h, w, c]
        // output -> [h', w', n]

        let startx = 0 - (self.pad.x as i32);
        let starty = 0 - (self.pad.y as i32);

        let endy = (self.h as i32) - (self.kernel_size as i32) + (self.pad.y as i32);
        assert!(endy > 0 && endy > starty);

        let endx = (self.h as i32) - (self.kernel_size as i32) + (self.pad.y as i32);
        assert!(endx > 0 && endx > startx);

        let batch = state.net.batch;

        for filter in 0..self.n {
            // for each filter
            for channel in 0..self.c {
                let mut y = starty;
                while y < endy {
                    // for each input channel in conv

                    let mut x = startx;
                    while x < endx {
                        let mut kernel_sum = 0.;
                        for ky in 0..self.kernel_size {
                            let yloc = ky + y as usize;
                            for kx in 0..self.kernel_size {
                                let xloc = kx + x as usize;

                                let px = {
                                    if xloc < 0 || yloc < 0{
                                        0.0
                                    }else{
                                        self.inputs[[batch, yloc, xloc, channel]]
                                    }
                                };
        
                                let w = self.weights[[ky, kx, channel, filter]];
                                kernel_sum += w * px + self.biases[filter];
                            }
                        }

                        self.outputs[[batch, y as usize, x as usize, filter]] = kernel_sum;
                        x += self.stride.x as i32;
                    }
                    y += self.stride.y as i32;
                }
            }
        }
        
        self.outputs
            .clone_from(&activate_array(&self.outputs.view(), &self.act_fn));

        state.input.clone_from(&self.outputs);
    }

    fn backward_conv(&mut self, state: &mut NetworkState) -> () {}
    fn update_conv(&mut self, state: &mut NetworkState) -> () {}
}
