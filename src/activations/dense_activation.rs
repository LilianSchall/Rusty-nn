use crate::maths::matrices::Matrix;
use super::LEAKY_RELU_VALUE;

use std::process;

pub enum DenseActivation {
    NoActivation, // for safety
    Sigmoid,
    Relu,
    LeakyRelu,
    Softmax, // output layer only
    Tanh
}

fn sigmoid(Z: &mut Matrix) {
    for i in 0..Z.values.len() {
        let minus_z = Z.values[i];
        Z.values[i] = 1.0 / (1.0 + minus_z.exp());
    }
}

fn relu(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = if Z.values[i] > 0.0 {Z.values[i]} else {0.0};
    }
}

fn leaky_relu(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = if Z.values[i] > 0.0 {Z.values[i] * LEAKY_RELU_VALUE} else {0.0};
    }
}

fn softmax(Z: &mut Matrix){

    let sum: f64 = Z.values.iter().sum();

    for i in 0..Z.values.len() {
        Z.values[i] = Z.values[i].exp() / sum;
    }
}

fn tanh(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = Z.values[i].tanh();
    }
}

pub fn apply_activation(activation: &DenseActivation, Z: &mut Matrix ) {
    match activation {
        DenseActivation::NoActivation =>  {
            println!("No Activation function, exiting..."); 
            process::exit(1);
        },
        DenseActivation::Sigmoid => sigmoid(Z),
        DenseActivation::Relu => relu(Z),
        DenseActivation::LeakyRelu => leaky_relu(Z),
        DenseActivation::Softmax => softmax(Z),
        DenseActivation::Tanh => tanh(Z)
    }
}

