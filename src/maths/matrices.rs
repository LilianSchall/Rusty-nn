use rand::Rng;
use std::process;

pub struct Matrix {
    pub x_length: usize,
    pub y_length: usize,
    pub values: Vec<f64>
}

impl Matrix {
    pub fn new(x_length: usize, y_length: usize) -> Matrix {

        let size: usize = x_length * y_length;

        let mut values = Vec::with_capacity(size);

        for _ in 0..size{
            values.push(0.0);
        } 

        Matrix {
            x_length: x_length,
            y_length: y_length,
            values: values
        }
    }

    pub fn print(&self) {
        for i in 0..self.y_length {
            for j in 0..self.x_length {
                let x = self.get(i, j);

                print!("{x}, ");
            }
            println!("");
        }
    }

    pub fn shuffle(mut self) -> Matrix {
        
        let mut rng = rand::thread_rng();

        for i in 0.. self.values.len() {
            self.values[i] = rng.gen::<f64>();
        }

        self
    }

    pub fn copy(&self) -> Matrix {
        let mut copied: Matrix = Matrix::new(self.x_length, self.y_length);

        for i in 0..self.values.len() {
            copied.values[i] = self.values[i];
        }

        copied
    }

    pub fn get(&self, y: usize, x: usize) -> f64 {
        if y >= self.y_length || x >= self.x_length {
            let ymax = self.y_length;
            let xmax = self.x_length;

            println!("Error: GET method for matrix has encountered an exception");
            println!("Expected to have a y or a x lower then y_length or x_length");
            println!("--------DEBUG------------");
            println!("y: {y}, x: {x}, |");
            println!("y_length: {ymax}, x_length: {xmax}");
            process::exit(1);
        }
        self.values[y * self.x_length + x]
    }

    pub fn set(&mut self, y: usize, x: usize, value: f64) {
        if y >= self.y_length || x >= self.x_length {
            let ymax = self.y_length;
            let xmax = self.x_length;

            println!("Error: SET method for matrix has encountered an exception");
            println!("Expected to have a y or a x lower then y_length or x_length");
            println!("--------DEBUG------------");
            println!("y: {y}, x: {x}");
            println!("y_length: {ymax}, x_length: {xmax}");
            process::exit(1);
        }

        self.values[y * self.x_length + x] = value;
    }

    pub fn dot(mat1: &Matrix, mat2: &Matrix) -> Matrix {
        if mat1.x_length != mat2.y_length {
            let mat1x = mat1.x_length;
            let mat2y = mat2.y_length;

            println!("Error: DOT function for matrix has encountered an exception");
            println!("Expected to have a two matrices of y size = x size");
            println!("--------DEBUG------------");
            println!("mat1x: {mat1x}, mat2y: {mat2y}");
            process::exit(1);
        }
        let mut mat = Matrix::new(mat2.x_length, mat1.y_length);
        
        for y in 0..mat.y_length {
            for x in 0..mat.x_length {
                let mut value: f64= 0.0;
                for n in 0..mat1.x_length { // or mat2.y_length 
                    
                    value += mat1.get(y,n) * mat2.get(n,x);
                }
                mat.set(y, x, value);
            }
        }

        mat

    }

    pub fn add(mat1: &Matrix, mat2: &Matrix) -> Matrix {
        if mat1.x_length != mat2.x_length || mat1.y_length != mat2.y_length {
            let mat1x = mat1.x_length;
            let mat1y = mat1.y_length;

            let mat2x = mat2.x_length;
            let mat2y = mat2.y_length;

            println!("Error: ADD function for matrix has encountered an exception");
            println!("Expected to have a two matrices of same size");
            println!("--------DEBUG------------");
            println!("mat1x: {mat1x}, mat1y: {mat1y}");
            println!("mat2x: {mat2x}, mat2y: {mat2y}");

            process::exit(1);
        }

        let mut mat = Matrix::new(mat1.x_length, mat1.y_length);

        for y in 0..mat.y_length {
            for x in 0..mat.x_length { 
                mat.set(y, x, mat1.get(y,x) + mat2.get(y, x));
            }
        }

        mat
    }

    pub fn vec_to_col_mat(vec: &Vec<f64>) -> Matrix{
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