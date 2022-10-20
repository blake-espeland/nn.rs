use crate::blas::*;
use crate::nn::LearningRatePolicy;
use crate::nn::convolutional_layer::ConvolutionalLayer;
use crate::nn::layer::Layer;
use crate::nn::network::Network;
use crate::nn::Activation;
use crate::nn::network::UpdateArgs;
use crate::util::dtypes::*;

use json::stringify;
use json::JsonValue;
use json::JsonValue::Array as JArray;
use json::JsonValue::String as JString;

use std::env;
use std::fs;

fn json_val_to_string(j: &JsonValue) -> String {
    String::from(j.as_str().unwrap_or(""))
}

fn get_float(j: &JsonValue, key: &str) -> Float {
    j[key].as_f32().expect("Unable to unpack option to float.")
}

fn get_usize(j: &JsonValue, key: &str) -> usize {
    j[key]
        .as_usize()
        .expect("Unable to unpack option to usize.")
}

fn get_Two(j: &JsonValue, key: &str) -> Two<usize> {
    match j[key] {
        JArray(a) => {
            if a.len() == 2 {
                Two {
                    x: a[0].as_usize().expect("Unable to unpack option to usize."),
                    y: a[1].as_usize().expect("Unable to unpack option to usize."),
                }
            } else {
                return Two { x: 0, y: 0 };
            }
        }
        _ => return Two { x: 0, y: 0 },
    }
}

fn get_act_fn(j: &JsonValue, key: &str) -> Activation {
    match &j[key] {
        JString(x) => match x.as_str() {
            "Relu" => Activation::Relu,
            "ID" => Activation::ID,
            &_ => Activation::Relu
        },
        _ => Activation::Relu,
    }
}

fn get_policy(j: &JsonValue, key: &str) -> LearningRatePolicy {
    match j[key] {
        JString(x) => match x.as_str() {
            "Constant" => LearningRatePolicy::Constant,
            "Step" => LearningRatePolicy::Step,
            "Poly" => LearningRatePolicy::Poly,
            "Steps" => LearningRatePolicy::Steps,
            "Sig" => LearningRatePolicy::Sig,
            "Random" => LearningRatePolicy::Random,
            &_ => LearningRatePolicy::Steps
        },
        _ => LearningRatePolicy::Steps
    }
} 

fn get_vec<T: std::convert::From<T>>(j: &JsonValue, key: &str) -> Vec<T> {
    match j[key] {
        JArray(a) => {
            a.into_iter().map(|item| T::from(item))
        },
        _ => Vec::default()
    }
}

pub fn parse_cfg(path_cfg: &str) -> Network {
    println!("Reading model config from: {}", path_cfg);

    // Read config file from filesystem
    let raw_cfg = fs::read_to_string(path_cfg).expect("Unable to read file");

    // Parsing json
    let json_cfg = json::parse(raw_cfg.as_str()).expect("Unable to parse json");

    let layers = &json_cfg["Layers"];
    let hyps = &json_cfg["Hyperparams"];

    assert!(!layers.is_null(), "Must contain \"Layers\" section on cfg");
    assert!(
        !hyps.is_null(),
        "Must contain \"Hyperparams\" section on cfg"
    );

    let batch = get_usize(hyps, "Batch");
    let subdiv = get_usize(hyps, "Subdivisions");
    let t = get_usize(hyps, "TimeSteps");

    let ua = 
    UpdateArgs {
        cbatch: batch / subdiv,
        lr: get_float(hyps, "LearningRate"),
        adam: get_usize(hyps, "Adam"),
        momentum: get_float(hyps, "Momentum"),
        decay: get_float(hyps, "Decay"),
        B1: 0.0,
        B2: 0.0,
        eps: 0.00001,
        t: t
    };

    let mut net: Network = Network::new(
        batch,
        subdiv,
        1,
        1,
        t,
        get_usize(hyps, "Height"),
        get_usize(hyps, "Width"),
        get_usize(hyps, "Channels"),
        get_policy(hyps, "Policy"),
        ua,

    );

    net.batch_size = get_usize(hyps, "Batch");
    net.subdivisions = get_usize(hyps, "Subdivisions");
    net.t = get_usize(hyps, "TimeSteps");
    net.h = get_usize(hyps, "Height");
    net.w = get_usize(hyps, "Width");
    net.c = get_usize(hyps, "Channels");

    let mut i = 0;
    let mut ishape = DataShape {
        b: net.batch_size,
        h: net.h,
        w: net.w,
        c: net.c,
    };
    let mut oshape = DataShape::default();

    for layer in layers.members() {
        if layer["Type"] == "Connected" {}
        if layer["Type"] == "Conv" {
            println!("{}: Conv", i);
            let k = get_usize(layer, "Size");
            let n = get_usize(layer, "Filters");
            let a = get_act_fn(layer, "Activation");
            let s = get_Two(layer, "Stride");
            let p = get_Two(layer, "Pad");

            oshape = 
                DataShape {
                    b: net.batch_size,
                    h: convolutional_out_width(ishape.w, k, &s, &p),
                    w: convolutional_out_height(ishape.h, k, &s, &p),
                    c: n
                };

            net.add_layer(Layer::new_conv(
                0, net.t, k, a, ishape, oshape, &p, &s,
            ))
        }
        i += 1;
        ishape = oshape;
    }

    net
}
