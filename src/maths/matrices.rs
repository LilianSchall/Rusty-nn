pub struct Matrix {
    x_length: usize,
    y_length: usize,
    values: Vec<f64>
}

impl Matrix {
    pub fn new(x_length: usize, y_length: usize) -> Matrix {

        let values = Vec::with_capacity(x_length * y_length);

        Matrix {
            x_length: x_length,
            y_length: y_length,
            values: values
        }
    }

    
}