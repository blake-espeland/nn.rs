use super::traits::FCLayerTrait;
use crate::util::dtypes::CFloat;
use crate::blas::mult::*;

use ndarray::{Array2, Array1, ArrayView1};

use crate::nn::activation::*;

#[derive(Clone)]
pub struct FCLayer{
    pub is_input: bool,
    pub is_output: bool,
    pub act: act_t,
    pub act_b: act_t,
    pub w: Array2<CFloat>, // [n x m] n = cur nodes, m = next nodes
}

impl FCLayerTrait for FCLayer{
    fn forward(&self, x: &ArrayView1<CFloat>) -> Array1<CFloat> {
        let new = mult2by1(&self.w.view(), x).unwrap();
        (self.act)(new.view())
    }

    // Needs to be modified
    fn backward(&self, x: &ArrayView1<CFloat>) -> Array1<CFloat> {
        let new = mult2by1(&self.w.view(), x).unwrap();
        (self.act)(new.view())
    }

    fn new(i: usize, o: usize, a: ACT, is_input: bool, is_output: bool) -> FCLayer{
        FCLayer { 
            is_input: is_input,
            is_output: is_output, 
            act: get_act_f(&a), 
            act_b: get_act_b(&a), 
            w: Array2::<CFloat>::ones((i, o)) 
        }
    }
}




impl Default for FCLayer{
    fn default() -> FCLayer{
        FCLayer {
            is_input: false,
            is_output: false,
            act: id_f,
            act_b: id_b,
            w: Array2::<CFloat>::default((0, 0))
        }
    }
}