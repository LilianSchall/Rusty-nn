use crate::activations::{dense_activation, LEAKY_RELU_VALUE};
use dense_activation::DenseActivation;

use std::process;


pub fn d_sigmoid(x: f64) -> f64 {
    dense_activation::__sigmoid(x) * (1.0 - dense_activation::__sigmoid(x))
}

pub fn d_relu(x: f64) -> f64 {
    if x < 0.0 {0.0} else {1.0}
}

pub fn d_lealy_relu(x: f64) -> f64 {
    if x < 0.0 {0.0} else {LEAKY_RELU_VALUE}
}

pub fn d_tanh(x: f64) -> f64 {
    let y: f64 = x.tanh();

    1.0 - y.powi(2)
}

pub fn apply_derivation(activation: &DenseActivation, x: f64) -> f64 {

    match activation {
        DenseActivation::NoActivation =>  {
            println!("No Activation function, exiting..."); 
            process::exit(1);
        },
        DenseActivation::Sigmoid => d_sigmoid(x),
        DenseActivation::Relu => d_relu(x),
        DenseActivation::LeakyRelu => d_lealy_relu(x),
        DenseActivation::Tanh => d_tanh(x),
        DenseActivation::Softmax => d_sigmoid(x),
    }
}