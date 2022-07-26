use crate::activations::dense_activation::DenseActivation;
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
    values: Vec<Matrix> 
}

impl DenseModel {
    pub fn new(activations_arr: Vec<DenseActivation>, loss: DenseLosses,
    shapes: Vec<DenseShape>) -> DenseModel {
        
        let mut values = Vec::with_capacity(activations_arr.len());
        let mut weights = Vec::with_capacity(activations_arr.len());
        let biases = Vec::with_capacity(activations_arr.len());

        let length = activations_arr.len() - 1;

        for i in 0..length {
            values[i] = Matrix::new(1,shapes[i].range);
            weights[i] = Matrix::new(shapes[i].range, shapes[i + 1].range);
        }
        values[length] = Matrix::new(1,shapes[length].range);

        

        DenseModel {
            nb_layers: activations_arr.len(),
            loss: loss,
            activations: activations_arr,
            weights: weights,
            biases: biases,
            values: values
        }
    }
}