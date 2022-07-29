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

fn sigmoid(mat: &mut Matrix) {
    for i in 0..mat.values.len() {
        mat.values[i] = __sigmoid(mat.values[i]);
    }
}

pub fn __relu(x: f64) -> f64 {
    if x > 0.0 {x} else {0.0}
}

fn relu(mat: &mut Matrix){
    for i in 0..mat.values.len() {
        mat.values[i] = __relu(mat.values[i]);
    }
}

pub fn __leaky_relu(x: f64) -> f64 {
    if x > 0.0 {x * LEAKY_RELU_VALUE} else {0.0}
}

fn leaky_relu(mat: &mut Matrix){
    for i in 0..mat.values.len() {
        mat.values[i] = __leaky_relu(mat.values[i]);
    }
}

fn softmax(mat: &mut Matrix){

    let sum: f64 = mat.values.iter().sum();

    for i in 0..mat.values.len() {
        mat.values[i] = mat.values[i].exp() / sum;
    }
}


pub fn __tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh(mat: &mut Matrix){
    for i in 0..mat.values.len() {
        mat.values[i] = __tanh(mat.values[i]);
    }
}

pub fn apply_activation(activation: &DenseActivation, mat: &mut Matrix ) {
    match activation {
        DenseActivation::NoActivation =>  {
            println!("No Activation function, exiting..."); 
            process::exit(1);
        },
        DenseActivation::Sigmoid => sigmoid(mat),
        DenseActivation::Relu => relu(mat),
        DenseActivation::LeakyRelu => leaky_relu(mat),
        DenseActivation::Softmax => softmax(mat),
        DenseActivation::Tanh => tanh(mat)
    }
}

