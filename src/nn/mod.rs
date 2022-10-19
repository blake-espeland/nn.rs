pub mod activation;
pub mod layer;
pub mod state;
pub mod network;

pub enum WeightsTypeT{
    NoWeights, PerFeature, PerChannel
}

pub enum WeightsNormalizationT{
    NoNormalization, ReluNormalization, SoftmaxNormalization
}


#[allow(non_camel_case_types)]
pub enum Activation{
    ID,
    Relu,
    Mish,
    Swish,
    Logistic
}


pub enum LearningRatePolicy{
    Constant,
    Step,
    Poly,
    Steps,
    Sig,
    Random
}