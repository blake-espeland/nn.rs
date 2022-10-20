use crate::nn::network::Network;

use json::JsonValue;
use json::stringify;

use std::env;
use std::fs;

fn json_val_to_string(j: &JsonValue) -> String {
    String::from(j.as_str().unwrap_or(""))
}

pub fn parse_cfg(path_cfg: &str) -> Network{
    println!("Reading model config from: {}", path_cfg);

    // Read config file from filesystem
    let raw_cfg = fs::read_to_string(path_cfg).expect("Unable to read file");

    // Parsing json
    let json_cfg = json::parse(raw_cfg.as_str()).expect("Unable to parse json");   

    let layers = &json_cfg["Layers"];
    let hyps = &json_cfg["Hyperparams"];

    assert!(!layers.is_null(), "Must contain \"Layers\" section on cfg");
    assert!(!hyps.is_null(), "Must contain \"Hyperparams\" section on cfg");

    let mut net: Network = Network::default();

    let mut i = 0;
    let mut cur_shape = 0;
    let mut prev_shape = 0;
    
    for layer in layers.members(){
        if layer["Type"] == "Connected" {
            println!("{}: Connected", i);
        }
        if layer["Type"] == "Conv" {
            println!("{}: Conv", i);

        }
        i += 1;
    }
    net
}

