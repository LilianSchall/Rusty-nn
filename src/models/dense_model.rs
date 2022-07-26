use crate::activations::dense_activation::DenseActivation;
use crate::losses::dense_losses::DenseLosses;

pub struct DenseModel {
    nb_layers: usize,
    loss: DenseLosses,
    activations: Vec<DenseActivation>
}

impl DenseModel {
    pub fn new(activations_arr: Vec<DenseActivation>, loss: DenseLosses)
    -> DenseModel {
        
        DenseModel {
            nb_layers: activations_arr.len(),
            loss: loss,
            activations: activations_arr
        }
    }
}