use crate::util::dtypes::Float;
use super::layer::Layer;
use super::state::NetworkState;

use ndarray::{Array, ArrayView, arr1, Ix, ArrayView1, Array1};

pub type act_t = fn(l: &mut Layer, s: &mut NetworkState) -> ();
pub type grad_t = fn(x: Float) -> Float;

#[allow(non_camel_case_types)]
pub enum ACTIVATION{
    ID,
    RELU,
    MISH,
    SWISH,
    LOGISTIC
}

pub fn get_act_f(a: &ACTIVATION) -> act_t{
    match a {
        ID => {id}
        RELU => {relu}
        _ => {relu}
    }
}

pub fn get_gradient(a: &ACTIVATION) -> grad_t{
    match a {
        ID => {id_grad}
        RELU => {relu_grad}
        _ => {id_grad}
    }
}


/*

Applies activation to an array of values
    x -> data to be activated on
    d -> dimensionality of data
*/ 
pub fn activate_array<D>(x: &mut Array<Float, D>, d: Ix, a: ACTIVATION) -> (){
    
}

pub fn relu (l: &mut Layer, s: &mut NetworkState) -> (){}
pub fn relu_grad(x: Float) -> Float { if x > 0. { 1. } else { 0. } }

pub fn id (l: &mut Layer, s: &mut NetworkState) -> () {}
pub fn id_grad (x: Float) -> Float { x }