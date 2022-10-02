use super::config as config;
use std::env;
use std::fs;


pub fn parse_args() -> config::Config{

    let cla: Vec<String> = env::args().collect();
    let mut cfg: config::Config = config::Config::default();

    assert!(cla.len() == 2, "Missing argument: config path");
    cfg.cfg_path = cla[1].to_owned();

    let yaml_contents = fs::read_to_string(&cfg.cfg_path)
        .expect(format!("Could not read file {}", cfg.cfg_path).as_str());

    println!("{:?}", yaml_contents);

    cfg
}
