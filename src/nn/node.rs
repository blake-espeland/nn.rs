use super::connection::{*};

pub type Nval = f32;

pub struct Node{
    val: Nval,
    forward: Vec<Connection>,
    backward: Vec<Connection>
}

impl Node{
    pub fn update(&mut self, new: Nval){
        self.val = new;
    }
}

impl Default for Node{
    fn default() -> Node{
        Node{
            val: 0.0
        }
    }
}