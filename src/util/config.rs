use crate::tensor::shape as shape;
use shape::DimT as DimT;

pub struct Config{
    cfg_path: String,
    input_shape: shape::Shape,
    output_shape: shape::Shape,
    n_layers: i32,
}

impl Config{
    pub fn set_ishape(c: &mut Config, i: &Vec<DimT>) -> (){
        c.input_shape.set_dims(i);
    }
}