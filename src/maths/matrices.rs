pub struct Matrix {
    pub x_length: usize,
    pub y_length: usize,
    pub values: Vec<f64>
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

    pub fn copy(&self) -> Matrix {
        let mut copied: Matrix = Matrix::new(self.x_length, self.y_length);

        for i in 0..self.values.len() {
            copied.values[i] = self.values[i];
        }

        copied
    }

    pub fn get(&self, y: usize, x: usize) -> f64 {
        self.values[y * self.x_length + x]
    }

    pub fn set(&mut self, y: usize, x: usize, value: f64) {
        if y >= self.y_length || x >= self.x_length {
            return;
        }

        self.values[y * self.x_length + x] = value;
    }

    pub fn dot(mat1: &Matrix, mat2: &Matrix) -> Option<Matrix> {
        if mat1.x_length != mat2.y_length {
            return None;
        }
        let mut Z = Matrix::new(mat2.x_length, mat1.y_length);
        
        for y in 0..Z.y_length {
            for x in 0..Z.x_length {
                let mut value: f64= 0.0;
                for n in 0..mat1.x_length { // or mat2.y_length 
                    
                    value += mat1.get(y,n) * mat2.get(n,x);
                }
                Z.set(y, x, value);
            }
        }

        Some(Z)

    }

    pub fn add(mat1: &Matrix, mat2: &Matrix) -> Option<Matrix> {
        if mat1.x_length != mat2.x_length || mat1.y_length != mat2.y_length {
            return None;
        }

        let mut Z = Matrix::new(mat1.x_length, mat1.y_length);

        for y in 0..Z.y_length {
            for x in 0..Z.x_length { 
                
                Z.set(y, x, mat1.get(y,x) + mat2.get(y, x));
            }
        }

        Some(Z)
    }

    pub fn vec_to_col_mat(vec: Vec<f64>) -> Matrix{
        let mut result = Matrix::new(1, vec.len());

        for i in 0..vec.len() {
            result.values[i] = vec[i];
        }

        result
    }

    pub fn add_real(&mut self, b: f64) {
        for i in 0..self.values.len() {
            self.values[i] += b;
        }
    }
    
}