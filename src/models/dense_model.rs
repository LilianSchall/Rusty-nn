use crate::activations::dense_activation::{apply_activation, DenseActivation};
use crate::derivations::dense_derivation::{apply_derivation};
use crate::losses::dense_losses::{DenseLosses, derivative_error};
use crate::shapes::dense_shape::DenseShape;
use crate::maths::matrices::Matrix;

pub struct DenseModel {
    nb_layers: usize,
    loss: DenseLosses,
    activations: Vec<DenseActivation>,

    // three-dimensional vector:
    // z selects the group of weights between two layers
    weights: Vec<Matrix>,

    // two-dimensional vector (Matrix is in L(1,n))
    // each perceptron of layer has a certain bias attributed to it.
    biases: Vec<Matrix>,

    // two-dimensional vector (Matrix is in L(1,n))
    raw_values: Vec<Matrix>,

    // two-dimensional vector (Matrix is in L(1,n))
    values: Vec<Matrix> 
}

impl DenseModel {
    pub fn new(activations_arr: Vec<DenseActivation>, loss: DenseLosses,
    shapes: Vec<DenseShape>) -> DenseModel {
        
        let mut values = Vec::with_capacity(shapes.len());
        let mut raw_values = Vec::with_capacity(shapes.len());

        let mut weights = Vec::with_capacity(activations_arr.len()); //shapes.len() - 1
        let mut biases = Vec::with_capacity(activations_arr.len()); //shapes.len() - 1

        let length = activations_arr.len();

        for i in 0..length {
            values.push(Matrix::new(1,shapes[i].range));
            raw_values.push(Matrix::new(1,shapes[i].range).shuffle());

            weights.push(Matrix::new(shapes[i].range, shapes[i + 1].range).shuffle());
            biases.push(Matrix::new(1, shapes[i + 1].range).shuffle());
        }
        values.push(Matrix::new(1,shapes[length].range));
        raw_values.push(Matrix::new(1,shapes[length].range));

        DenseModel {
            nb_layers: shapes.len(),
            loss: loss,
            activations: activations_arr,
            weights: weights,
            biases: biases,
            raw_values: raw_values,
            values: values
        }
    }

    pub fn feed_forward(&mut self, input: Vec<f64>) {
        if input.len() != self.values[0].y_length {
            return;
        }

        self.values[0] = Matrix::vec_to_col_mat(input);
        self.raw_values[0] = self.values[0].copy();

        for i in 0..(self.nb_layers - 1) {

            let mut mat = Matrix::dot(&self.weights[i], &self.values[i]);

            mat = Matrix::add(&mat, &self.biases[i]);

            self.raw_values[i + 1] = mat.copy();

            apply_activation(&self.activations[i], &mut mat);
            self.values[i + 1] = mat;
        }
    }

    pub fn back_propagate(&mut self, output: &Matrix) -> Vec<Vec<f64>> {
        
        let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(self.nb_layers);
        for _ in 0..self.nb_layers {
            deltas.push(Vec::new());
        }

        for l in (1..self.nb_layers).rev() {
            for i in 0..self.values[l].y_length {
                let d_activation = apply_derivation(&self.activations[l - 1], self.raw_values[l].get(i,0));
                let result: f64;
                if l == self.nb_layers - 1 {
                    let d_cost = derivative_error(&self.loss, self.values[l].y_length, self.values[l].get(i,0), output.get(i,0));
                    result = d_activation * d_cost;
                }
                else {

                    let mut sum: f64 = 0.0;

                    for j in 0..deltas[l + 1].len() {
                        // why weights[l] and not l + 1 ?
                        // Well, it is for the unique reason that self.weights.len() = self.nb_layers - 1
                        sum += deltas[l + 1][j] * self.weights[l].get(j,i);
                    }
                    result = sum * d_activation;
                }
            
                deltas[l].push(result);
            }
            deltas[l].reverse();
        }
        deltas
    }

    pub fn update_weights(&mut self, deltas: &Vec<Vec<f64>>, learning_rate: f64) {
        for l in (1..self.nb_layers).rev() {
            for i in 0..deltas[l].len() {
                
                let derivative_wrt_bias: f64 = deltas[l][i];

                for j in 0..self.values[l - 1].y_length {

                    let derivative_wrt_weight: f64 = self.values[l - 1].get(j,0) * deltas[l][i];
                    let new_weight_value: f64 = self.weights[l - 1].get(i, j) - derivative_wrt_weight * learning_rate;
                    self.weights[l - 1].set(i, j, new_weight_value); 
                }
                self.biases[l - 1].set(i,0, derivative_wrt_bias);
            }
        }
    }
}