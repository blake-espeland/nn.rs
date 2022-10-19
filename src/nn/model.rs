use super::layers::traits::FCLayerTrait;
use crate::nn::activation::ACT;

pub struct FCModel<M: FCLayerTrait + Default> {
    output: M,
    hidden: Vec<M>
}

impl<M: FCLayerTrait + Default> FCModel<M> {
    fn new(i: usize, o: usize, h: &Vec<M>) -> FCModel<M>{
        FCModel {
            output: M::new(i, o, ACT::ID, false, true),
            hidden: h.clone()
        }
    }
}

impl<M: FCLayerTrait + Default> Default for FCModel<M> {
    fn default() -> FCModel<M> {
        FCModel {
            output: M::default(),
            hidden: Vec::default()
        }
    }
}