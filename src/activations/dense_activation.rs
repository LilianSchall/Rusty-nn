use crate::models::dense_model::DenseModel;

pub enum DenseActivation {
    NoActivation, // for safety
    Sigmoid,
    Relu,
    LeakyRelu,
    Softmax, // output layer only
    Tanh
}

pub fn apply_activation(model: DenseModel) {
    
}

fn sigmoid(z: f64) -> f64 {
    0.0
}

fn relu(z: f64) -> f64 {
    0.0
}

fn leaky_relu(z: f64) -> f64 {
    0.0
}

fn softmax(z: f64) -> f64 {
    0.0
}

fn tanh(z: f64) -> f64 {
    0.0
}