use super::config as config;
use super::config::Config;
use crate::nn::network::Network;

use json::JsonValue;
use json::stringify;

use std::env;
use std::fs;

fn json_val_to_string(j: &JsonValue) -> String {
    String::from(j.as_str().unwrap_or(""))
}

pub fn parse_cfg(path: &str){
    println!("Reading model config from: {}", path);

    // Read config file from filesystem
    let raw = fs::read_to_string(path).expect("Unable to read file");

    // Parsing json
    let json = json::parse(raw.as_str()).expect("Unable to parse json");
    let layers = &json["Layers"];

    assert!(!layers.is_null(), "Must contain \"Layers\" section on cfg");

    let mut layer_v: Vec<JsonValue> = Vec::<JsonValue>::new();

    for layer in layers.members(){
        layer_v.push(layer.clone());
    }
}

impl Network {

}

