pub mod activations;
pub mod data;
pub mod derivations;
pub mod losses;
pub mod maths;
pub mod models;
pub mod shapes;

use data::create_data::{load_data, Sample};
use maths::matrices::Matrix;
use models::dense_model::DenseModel;
use losses::dense_losses::DenseLosses;
use activations::dense_activation::DenseActivation;
use shapes::dense_shape::DenseShape;

fn main() {

    /*let mut activations_architecture = Vec::with_capacity(3);
    let loss: DenseLosses = DenseLosses::CategoricalCrossEntropy;
    let mut shape: Vec<DenseShape> = Vec::with_capacity(4);

    shape.push(DenseShape::new(2, 1, 1));
    shape.push(DenseShape::new(4, 1, 1));
    shape.push(DenseShape::new(3, 1, 1));
    shape.push(DenseShape::new(1, 1, 1));

    activations_architecture.push(DenseActivation::Sigmoid);
    activations_architecture.push(DenseActivation::Sigmoid);
    activations_architecture.push(DenseActivation::Sigmoid);

    let mut model: DenseModel = DenseModel::new(activations_architecture,loss, shape);

    let input: Vec<f64> = [1.0, 0.0].to_vec();
    let output: Matrix = Matrix::vec_to_col_mat([1.0].to_vec());
    

    model.feed_forward(input);
    let deltas = model.back_propagate(&output);

    model.update_weights(&deltas, 0.001);*/

    let samples = load_data(&"input.txt".to_string(), &"output.txt".to_string());

    for i in 0..samples.len() {
        samples[i].print_sample();
    }
}