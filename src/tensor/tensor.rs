use super::shape::Shape;

use std::ops::{Add, Mul};
use std::fmt::Display;
use std::fmt::Debug;

extern crate num;
use num::traits::Zero;
pub trait TensorTrait<T>: Zero + Clone + Display + Debug + Add<T, Output = T> + Mul<T, Output = T>{ }
impl<T> TensorTrait<T> for T where T: Zero + Clone + Display + Debug + Add<T, Output = T> + Mul<T, Output = T> { }

pub struct Tensor<T: TensorTrait<T>>{
    shape: Shape,
    data: Vec<T>
}


impl<T: TensorTrait<T>> Tensor<T> {
    fn build(s: &Shape, d: Vec<T>) -> Self{
        Tensor { shape: s.clone(), data: d }
    }
}

impl<T: TensorTrait<T>> Default for Tensor<T>{
    fn default() -> Self {
        Self { shape: Shape::default(), data: Vec::default() }
    }
}