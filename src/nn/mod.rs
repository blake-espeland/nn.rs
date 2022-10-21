pub mod activation;
pub mod layer;
pub mod state;
pub mod network;

pub mod convolutional_layer;
pub mod connected_layer;

pub enum WeightsTypeT{
    NoWeights, PerFeature, PerChannel
}

pub enum WeightsNormalizationT{
    NoNormalization, ReluNormalization, SoftmaxNormalization
}


#[derive(Clone, Copy)]
pub enum Activation{
    ID,
    Relu,
    Mish,
    Swish,
    Logistic
}


pub fn get_act_str(a: Activation) -> &'static str {
    match a {
        Activation::ID => { "Identity" },
        Activation::Relu => { "ReLu" },
        Activation::Mish => { "Mish" },
        Activation::Swish => { "Swish" },
        Activation::Logistic => { "Logistic" }
    }
}


pub enum LearningRatePolicy{
    Constant,
    Step,
    Poly,
    Steps,
    Sig,
    Random
}

pub enum LayerType{
    Connected, Conv, Normalization, Cost
}

pub enum CostType{
    SSE, Masked, L1, Seg, Smooth, WGAN
}