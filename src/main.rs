mod util;
mod nn;
mod blas;

fn main() {
    util::parser::parse_cfg("cfg/model.json");
}