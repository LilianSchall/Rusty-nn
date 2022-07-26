pub struct DenseShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub range: u32,
}

impl DenseShape {
    pub fn new(x: u32, y: u32, z: u32) -> DenseShape {

        DenseShape {
            x: x,
            y: y,
            z: z,
            range: x*y*z
        }
    }
    
}