use crate::tensor::shape as shape;

pub struct Config{
    pub cfg_path: String,
    pub net_type: String,
    pub input_shape: shape::Shape,
    pub output_shape: shape::Shape,
    pub n_layers: i32,
}

impl Default for Config{
    fn default() -> Config{
        Config{
            cfg_path: String::new(), 
            net_type: String::new(),
            input_shape: shape::Shape::default(), 
            output_shape: shape::Shape::default(),
            n_layers: 0
        }
    }
}