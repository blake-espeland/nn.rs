use ndarray::{ArrayView1, ArrayView2, Array1};
use crate::util::dtypes::Float;
use std::result::Result;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ArrayMultError;

impl fmt::Display for ArrayMultError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    	write!(f, "invalid first item to double")
    }
}

pub fn mult2by1(a: &ArrayView2<Float>, b: &ArrayView1<Float>) -> Result<Array1<Float>, ArrayMultError> {
    if (a.shape()[1] != b.len()){
        return Err(ArrayMultError)
    }

    let mut v = Array1::<Float>::zeros(a.shape()[0]);

    let mut i: usize = 0;
    let mut j: usize = 0;
    
    for aa in a.outer_iter(){
        j = 0;
        for bb in b.iter(){
            v[i] += aa[j] * bb;
            j += 1;
        }
        i += 1;
    }

    Ok(v)
}