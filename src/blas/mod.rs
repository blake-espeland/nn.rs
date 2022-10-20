use crate::nn::layer::Layer;
use crate::util::dtypes::*;

use ndarray::{Array, Array1, ArrayView1, ArrayView2, Dimension, Order, Shape};
use std::fmt;
use std::result::Result;

#[derive(Debug, Clone)]
pub struct ArrayMultError;
#[derive(Debug, Clone)]
pub struct ShapeCastError;

impl fmt::Display for ArrayMultError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to multiply Arrays")
    }
}

impl fmt::Display for ShapeCastError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to cast between shapes")
    }
}

pub fn mult2by1(
    a: &ArrayView2<Float>,
    b: &ArrayView1<Float>,
) -> Result<Array1<Float>, ArrayMultError> {
    if (a.shape()[1] != b.len()) {
        return Err(ArrayMultError);
    }

    let mut v = Array1::<Float>::zeros(a.shape()[0]);

    let mut i: usize = 0;
    let mut j: usize = 0;

    for aa in a.outer_iter() {
        j = 0;
        for bb in b.iter() {
            v[i] += aa[j] * bb;
            j += 1;
        }
        i += 1;
    }

    Ok(v)
}

pub fn flatten<D: Dimension>(x: &mut Array<Float, D>) -> Array1<Float> {
    let s = x.raw_dim().size();
    x.to_shape(s).unwrap().to_owned()
}

pub fn convolutional_out_height(
    h: usize,
    kernel_size: usize,
    stride: &Two<usize>,
    pad: &Two<usize>,
) -> usize {
    (h + 2 * pad.y - kernel_size) / stride.y + 1
}

pub fn convolutional_out_width(
    w: usize,
    kernel_size: usize,
    stride: &Two<usize>,
    pad: &Two<usize>,
) -> usize {
    (w + 2 * pad.x - kernel_size) / stride.x + 1
}
