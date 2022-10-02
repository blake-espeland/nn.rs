extern crate yaml_rust;

use super::config as config;
use std::env;
use std::fs;
use yaml_rust::YamlLoader;


pub fn parse_args() -> config::Config{

    let cla: Vec<String> = env::args().collect();
    let mut cfg: config::Config = config::Config::default();

    assert!(cla.len() == 2, "Missing argument: config path");
    cfg.cfg_path = cla[1].to_owned();

    let yaml_contents = fs::read_to_string(&cfg.cfg_path)
        .expect(format!("Could not read file {}", cfg.cfg_path).as_str());

    println!("{:?}", yaml_contents);

    let yaml = YamlLoader::load_from_str(yaml_contents.as_str()).unwrap()[0];

    cfg.input_shape = yaml["input_shape"].as_vec();
    cfg.output_shape = yaml["output_vector"].as_vec()
}
