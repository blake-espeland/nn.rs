use crate::util::dtypes::*;
use ndarray::{Array1, ArrayView1};
use super::super::activation::ACT;

pub trait FCLayerTrait: Clone{
    fn forward(&self, x: &ArrayView1<CFloat>) -> Array1<CFloat>;
    fn backward(&self, x: &ArrayView1<CFloat>) -> Array1<CFloat>;
    fn new(i: usize, o: usize, a: ACT, is_input: bool, is_output: bool) -> Self;
}