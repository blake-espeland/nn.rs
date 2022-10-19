
pub type DimT = i64;

#[derive(Clone)]
pub struct Shape{
    pub ndims: usize,
    pub dims: Vec<DimT>,
    pub size: DimT
}

impl Shape {
    pub fn calc_size(i: &Vec<DimT>) -> DimT{
        let mut s: DimT = 1;
        for x in i{
            s *= x;
        }
        s
    }

    pub fn set_dims(&mut self, i: &Vec<DimT>){
        self.size = Shape::calc_size(i);
        self.dims = i.to_owned();
        self.ndims = i.len();
    }
}

impl Default for Shape{
    fn default() -> Shape{
        Shape{ndims: 0, dims: Vec::default(), size: 0}
    }
}