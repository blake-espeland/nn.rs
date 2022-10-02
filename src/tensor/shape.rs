
pub type DimT = i64;

pub struct Shape{
    ndims: usize,
    dims: Vec<DimT>,
    size: DimT
}



impl Shape {
    pub fn calc_size(i: &Vec<DimT>) -> DimT{
        let mut s: DimT = 1;
        for x in i{
            s *= x;
        }
        s
    }

    pub fn make(i: &Vec<DimT>) -> Shape{
        Shape {ndims: i.len(), dims: i.to_owned(), size: Shape::calc_size(i)}
    }

    pub fn size(&self) -> &DimT{ &self.size }

    pub fn dims(&self) -> &Vec<DimT>{ &self.dims }
    pub fn set_dims(&mut self, i: &Vec<DimT>){
        self.size = Shape::calc_size(i);
        self.dims = i.to_owned();
        self.ndims = i.len();
    }

    pub fn ndims(&self) -> &usize { &self.ndims }
}