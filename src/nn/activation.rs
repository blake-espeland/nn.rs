
pub enum Activation{
    ID,
    RELU,
    MISH,
    SWISH,
    LOGISTIC
}

pub fn relu_f (x: c_float) -> c_float{ if x > 0 {x} else {0} }
pub fn relu_b (x: c_float) -> c_float{ x > 0 }

pub fn id_f (x: c_float){ x }
pub fn id_b (x: c_float) { 1.0 }