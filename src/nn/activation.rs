
pub enum Activation{
    RELU,
    MISH,
    SWISH,
    LOGISTIC
}

pub fn relu<T> (x: T) -> T{
    if x > 0 {x} else {0}
}

pub fn relu_back<T> (x: T) -> T{
    x > 0
}