use super::layer::Layer;
use super::state::NetworkState;
use super::Activation;
use crate::util::dtypes::{Array4F, Float};

use ndarray::{Array, Array1, ArrayView, ArrayView1, ArrayView4, Ix};

pub type ActFn = fn(x: Float) -> Float;
pub type GradFn = fn(x: Float) -> Float;

pub fn get_act_fn(a: &Activation) -> ActFn {
    match a {
        Activation::ID => id,
        Activation::Relu => relu,
        _ => relu,
    }
}

pub fn get_gradient_fn(a: &Activation) -> GradFn {
    match a {
        Activation::ID => id_grad,
        Activation::Relu => relu_grad,
        _ => id_grad,
    }
}

/*
Applies activation to an array of values
    x -> data to be activated on
    d -> dimensionality of data
*/
pub fn activate_array<'a>(xs: &ArrayView4<'a, Float>, a: &ActFn) -> Array4F {
    xs.map(|x| a(*x))
}

pub fn relu(x: Float) -> Float {
    if x > 0. {
        x
    } else {
        0.
    }
}
pub fn relu_grad(x: Float) -> Float {
    if x > 0. {
        1.
    } else {
        0.
    }
}

pub fn id(x: Float) -> Float { x }
pub fn id_grad(x: Float) -> Float { 1. }
