use crate::util::dtypes::CFloat;

use ndarray::{Array, Array1, ArrayView1};

pub type act_t = fn(x: ArrayView1<CFloat>) -> Array1<CFloat>;

pub enum ACT{
    ID,
    RELU,
    MISH,
    SWISH,
    LOGISTIC
}

pub fn get_act_f(a: &ACT) -> act_t{
    match a {
        ID => {id_f}
        RELU => {relu_f}
        _ => {relu_f}
    }
}

pub fn get_act_b(a: &ACT) -> act_t{
    match a {
        ID => {id_b}
        RELU => {relu_b}
        _ => {relu_b}
    }
}

fn _relu_f(x: &CFloat) -> CFloat { if x > &0. { *x } else { 0. } }
pub fn relu_f (x: ArrayView1<CFloat>) -> Array1<CFloat> { x.map(|_x| _relu_f(_x)) }
fn _relu_b(x: &CFloat) -> CFloat { if x > &0. { 1. } else { 0. } }
pub fn relu_b(x: ArrayView1<CFloat>) -> Array1<CFloat> { Array1::default(x.len()) }

pub fn id_f (x: ArrayView1<CFloat>) -> Array1<CFloat> { x.to_owned() }
pub fn id_b (x: ArrayView1<CFloat>) -> Array1<CFloat> { Array1::ones(x.len()) }