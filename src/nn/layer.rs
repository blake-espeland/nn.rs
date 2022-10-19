use ndarray::{Array1, Array2, Array3,
              ArrayView1, ArrayView2, ArrayView3};

use super::activation::*;
use crate::util::dtypes::*;

enum LayerType{
    Connected,
    Conv,
    Normalization,
    Cost
}

pub struct Layer {

    pub is_input: bool,
    pub is_output: bool,
    
    pub act: act_t,
    pub act_b: act_t,

    pub connected_weights: Array2<Float>, // [n x m] n = cur nodes, m = next nodes
}