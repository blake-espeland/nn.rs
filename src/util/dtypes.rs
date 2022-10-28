use ndarray::{Array1, Array2, Array3, Array4, ArrayView1};
use image::Rgb;

pub type Float = f32; // c-style float
pub type Int = i32;

pub type Array1F = Array1<Float>; // Assuming n truths per Y
pub type Array2F = Array2<Float>; // Assuming n truths per Y, and b Y's
pub type Array3F = Array3<Float>;
pub type Array4F = Array4<Float>;

#[derive(Clone, Copy)]
pub struct Two<T>{
    pub x: T,
    pub y: T
}

#[derive(Clone, Copy)]
pub struct Array4Shape {
    pub b: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl From<[usize; 4]> for Array4Shape{
    fn from(u: [usize; 4]) -> Self{
        Array4Shape { b: u[0], h: u[1], w: u[2], c: u[3] }
    }
}

impl Array4Shape {
    pub fn to_arr(&self) -> [usize; 4]{
        [self.b, self.h, self.w, self.c]
    }
}

impl Default for Array4Shape {
    fn default() -> Self {
        Array4Shape { b: 0, h: 0, w: 0, c: 0 }
    }
}

pub struct CVMat {
    pub shape: Array4Shape,

}