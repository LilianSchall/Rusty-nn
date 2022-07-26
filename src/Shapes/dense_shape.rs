pub struct DenseShape {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub range: usize,
}

impl DenseShape {
    pub fn new(x: usize, y: usize, z: usize) -> DenseShape {

        DenseShape {
            x: x,
            y: y,
            z: z,
            range: x*y*z
        }
    }
    
}