use crate::util::dtypes::*;

pub trait LayerTrait<T>: Clone + Debug{
    fn forward_pass(x: c_float) -> c_float;
    fn backward_pass();
    fn init_random();
}