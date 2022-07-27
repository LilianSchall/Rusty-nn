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

pub fn __sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid(Z: &mut Matrix) {
    for i in 0..Z.values.len() {
        Z.values[i] = __sigmoid(Z.values[i]);
    }
}

pub fn __relu(x: f64) -> f64 {
    if x > 0.0 {x} else {0.0}
}

fn relu(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = __relu(Z.values[i]);
    }
}

pub fn __leaky_relu(x: f64) -> f64 {
    if x > 0.0 {x * LEAKY_RELU_VALUE} else {0.0}
}

fn leaky_relu(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = __leaky_relu(Z.values[i]);
    }
}

fn softmax(Z: &mut Matrix){

    let sum: f64 = Z.values.iter().sum();

    for i in 0..Z.values.len() {
        Z.values[i] = Z.values[i].exp() / sum;
    }
}


pub fn __tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh(Z: &mut Matrix){
    for i in 0..Z.values.len() {
        Z.values[i] = __tanh(Z.values[i]);
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

