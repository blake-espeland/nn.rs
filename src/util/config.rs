use json::JsonValue;
use ndarray::Dim;

pub struct Config{
    pub cfg_path: String,
    
    pub n_layers: usize,

    // list of Strings that represent the different layers
    // ex. "FCLayer" -> FCLayer
    pub layers: Vec<JsonValue>
}

impl Default for Config{
    fn default() -> Config{
        Config{
            cfg_path: String::new(), 
            n_layers: 0,
            layers: Vec::<JsonValue>::new()
        }
    }
}