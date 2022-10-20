mod util;
mod nn;
mod blas;

fn main() {
    let mut net = util::parser::parse_cfg("cfg/model.json");
}