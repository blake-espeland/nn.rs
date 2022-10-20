use ndarray::{Array, IxDyn, Shape, Ix5};

pub type Float = f32; // c-style float
pub type Int = i32;
pub type Uchar = u8;

pub type FloatArr = Array<Float, IxDyn>;

#[derive(Clone, Copy)]
pub struct Two<T>{
    pub x: T,
    pub y: T
}

#[derive(Clone, Copy)]
pub struct Conv2dShape {
    pub b: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl From<[usize; 4]> for Conv2dShape{
    fn from(u: [usize; 4]) -> Self{
        Conv2dShape { b: u[0], h: u[1], w: u[2], c: u[3] }
    }
}

impl Conv2dShape {
    pub fn to_arr(&self) -> [usize; 4]{
        [self.b, self.h, self.w, self.c]
    }
}

