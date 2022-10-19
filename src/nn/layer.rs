use ndarray::{Array, Array1, Array2, Array3,
              ArrayView1, ArrayView2, ArrayView3, 
              Shape, Dimension, Dim, Ix};

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
    pub act_b: grad_t,

    pub input_shape: Shape<Ix>,
    pub output_shape: Shape<Ix>,
    pub connected_weights: Array<Float, Ix>, // [n x m] n = cur nodes, m = next nodes

    pub kernel: Array<Float, Ix>
}