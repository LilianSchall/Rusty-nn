use crate::activations::dense_activation::{apply_activation, DenseActivation};
use crate::losses::dense_losses::DenseLosses;
use crate::shapes::dense_shape::DenseShape;
use crate::maths::matrices::Matrix;

pub struct DenseModel {
    nb_layers: usize,
    loss: DenseLosses,
    activations: Vec<DenseActivation>,

    // three-dimensional vector:
    // z selects the group of weights between two layers
    weights: Vec<Matrix>,

    // this is an unidimensional vector:
    // each layer has a certain bias attributed to it.
    biases: Vec<f64>,

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
        let biases = Vec::with_capacity(activations_arr.len()); //shapes.len() - 1

        let length = activations_arr.len();

        for i in 0..length {
            values[i] = Matrix::new(1,shapes[i].range);
            raw_values[i] = Matrix::new(1,shapes[i].range);
            weights[i] = Matrix::new(shapes[i].range, shapes[i + 1].range);
        }
        values[length] = Matrix::new(1,shapes[length].range);

        

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

        for i in 0..(self.nb_layers - 1) {

            match Matrix::dot(&self.weights[i], &self.values[i]) {

                Some(mut Z) => {

                    Z.add_real(self.biases[i]);

                    self.raw_values[i + 1] = Z.copy();

                    apply_activation(&self.activations[i], &mut Z);

                    self.values[i + 1] = Z;
                }
                None => {}
            }

        }

    }
}