extern crate yaml_rust;

use super::config as config;
use crate::tensor::shape::DimT as DimT;

use std::env;
use std::fs;


use yaml_rust::YamlLoader as YamlLoader;
use yaml_rust::Yaml as Yaml;

fn yaml_arr_to_dim (y: &Vec<Yaml>) -> Vec<DimT>{
    y.into_iter().map(|x| x.as_i64().unwrap() as DimT).collect()
}

pub fn parse_args() -> config::Config{
    let REQ_YAML_FIELDS = ["input_shape", "output_shape", "type"];

    let cla: Vec<String> = env::args().collect();
    let mut cfg: config::Config = config::Config::default();

    assert!(cla.len() == 2, "Missing argument: config path");
    cfg.cfg_path = cla[1].to_owned();

    let yaml_contents = fs::read_to_string(&cfg.cfg_path)
        .expect(format!("Could not read file {}", cfg.cfg_path).as_str());
    let yaml = &YamlLoader::load_from_str(yaml_contents.as_str()).unwrap()[0];
    
    // for i in 0..REQ_YAML_FIELDS.len(){
    //     assert!(!yaml[REQ_YAML_FIELDS[i]].is_badvalue(), format!("{} required.", REQ_YAML_FIELDS[i]).as_str());
    // }
        
    let ipt_shp = yaml_arr_to_dim(yaml["input_shape"].as_vec().unwrap());
    let opt_shp = yaml_arr_to_dim(yaml["output_shape"].as_vec().unwrap());

    println!("Input shape: {:?}", ipt_shp);
    println!("Output shape: {:?}", opt_shp);

    cfg.input_shape.set_dims(&ipt_shp);
    cfg.output_shape.set_dims(&opt_shp);
    cfg.net_type = String::from(yaml["type"].as_str().unwrap_or_default());

    cfg
}
