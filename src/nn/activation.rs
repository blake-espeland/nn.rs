use crate::util::dtypes::Float;
use super::layer::Layer;
use super::state::NetworkState;
use super::Activation;

use ndarray::{Array, ArrayView, Ix, ArrayView1, Array1};

pub type ActFn = fn(l: &mut Layer, s: &mut NetworkState) -> ();
pub type GradFn = fn(x: Float) -> Float;



pub fn get_act_f(a: &Activation) -> ActFn{
    match a {
        Activation::ID => {id}
        Activation::Relu => {relu}
        _ => {relu}
    }
}

pub fn get_gradient(a: &Activation) -> GradFn{
    match a {
        Activation::ID => {id_grad}
        Activation::Relu => {relu_grad}
        _ => {id_grad}
    }
}


/*

Applies activation to an array of values
    x -> data to be activated on
    d -> dimensionality of data
*/ 
pub fn activate_array<D>(x: &mut Array<Float, D>, d: Ix, a: Activation) -> (){
    
}

pub fn relu (l: &mut Layer, s: &mut NetworkState) -> (){}
pub fn relu_grad(x: Float) -> Float { if x > 0. { 1. } else { 0. } }

pub fn id (l: &mut Layer, s: &mut NetworkState) -> () {}
pub fn id_grad (x: Float) -> Float { x }