pub mod activation;
pub mod layer;
pub mod state;
pub mod network;

pub mod convolutional_layer;
pub mod connected_layer;


const DefaultBatch: usize = 16;
const DefaultSubdiv: usize = 4;

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

#[derive(Clone)]
pub enum LayerType{
    Connected, Conv, Normalization, Cost
}

#[derive(Clone)]
pub enum CostType{
    SSE, Masked, L1, Seg, Smooth, WGAN
}