use super::connection::{*};

pub struct Node{
    val: c_float,
    forward: Vec<Connection>,
    backward: Vec<Connection>
}

impl Node{
    pub fn update(&mut self, new: c_float){
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