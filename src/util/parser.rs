use super::config as config;
use std::env;



pub fn parse_args() -> config::Config{
    let mut a: config::Config;
    let raw: Vec<String> = env::args().collect();

    let ipt_shape_str: String = String::new();
    let opt_shape_str: String = String::new();
    
    for arg in &raw{
        println!("{}", arg);
    }
    
    return a;
}
