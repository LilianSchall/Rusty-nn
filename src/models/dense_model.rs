use crate::activations::dense_activation::{apply_activation, DenseActivation};
use crate::derivations::dense_derivation::{apply_derivation};
use crate::losses::dense_losses::{DenseLosses, derivative_error};
use crate::shapes::dense_shape::DenseShape;
use crate::maths::matrices::Matrix;

use std::fs::File;
use std::io::Write;
use std::io::{BufReader, BufRead};
use std::str::FromStr;

pub struct DenseModel {
    nb_layers: usize,
    pub loss: DenseLosses,
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

    pub fn result(&self) -> Matrix{
        self.values[self.nb_layers - 1].copy()
    }

    pub fn feed_forward(&mut self, input: &Matrix) {
        if input.y_length != self.values[0].y_length {
            return;
        }
        self.values[0] = input.copy();
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
            for i in 0..self.weights[l - 1].y_length {
                let d_activation = apply_derivation(&self.activations[l - 1], self.raw_values[l].get(i,0));
                let result: f64;
                
                if l == self.nb_layers - 1 {
                    let d_cost = derivative_error(&self.loss, self.values[l].y_length, self.values[l].get(i,0), output.get(i,0));
                    result = d_activation * d_cost;
                }
                else {

                    let mut sum: f64 = 0.0;

                    for j in 0..self.weights[l].y_length {
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
            for i in 0..self.weights[l - 1].y_length {
                
                let derivative_wrt_bias: f64 = deltas[l][i];

                for j in 0..self.weights[l - 1].x_length {

                    let derivative_wrt_weight: f64 = self.values[l - 1].get(j,0) * deltas[l][i];
                    let new_weight_value: f64 = self.weights[l - 1].get(i, j) - derivative_wrt_weight * learning_rate;
                    self.weights[l - 1].set(i, j, new_weight_value); 
                }
                self.biases[l - 1].set(i,0, derivative_wrt_bias);
            }
        }
    }

    pub fn save(&self, filename: &String) {

        let mut archi_filename = filename.clone();
        let mut weights_filename = filename.clone();

        archi_filename.push_str(".arch");
        weights_filename.push_str(".wab");

        let mut archi_file: File = File::create(archi_filename).expect("Error while creating the file: {archi_filename}");
        let mut weights_file: File = File::create(weights_filename).expect("Error while creating the file: {weights_filename}");

        let structures: Vec<usize> = self.values.iter().map(|x| {x.y_length}).collect();

        let mut archi_content: String = String::new();

        archi_content.push_str(&structures.len().to_string());
        archi_content.push_str("\n");

        // saving the neurons and layers structure
        for i in 0..structures.len() {
            archi_content.push_str(&structures[i].to_string());

            if i != structures.len() - 1 {
                archi_content.push_str(" ");
            }
        }
        archi_content.push_str("\n");

        // saving the activations functions
        for i in 0..self.activations.len() {
            archi_content.push_str(&self.activations[i].to_string());

            if i != structures.len() - 1 {
                archi_content.push_str(" ");
            }
        }
        archi_content.push_str("\n");

        // saving the error / cost function
        archi_content.push_str(&self.loss.to_string());
        archi_content.push_str("\n");


        archi_file.write_all(archi_content.as_bytes()).expect("Error while saving the architecture of the model.");


        let mut weights_content: String = String::new();

        for l in 0..self.weights.len() {

            for i in 0..self.weights[l].y_length {

                for j in 0..self.weights[l].x_length {

                    weights_content.push_str(&self.weights[l].get(i,j).to_string());

                    if j != self.weights[l].x_length - 1 {
                        weights_content.push_str(" ");
                    }
                }
                weights_content.push_str("\n");
                weights_content.push_str(&self.biases[l].get(i,0).to_string());

                if l != self.weights.len() - 1 {
                    weights_content.push_str("\n");
                }
            }
        }

        weights_file.write_all(weights_content.as_bytes()).expect("Error while saving the weights and biases of the model.");
    }

    pub fn load_model(filename: &String) -> DenseModel {

        let mut archi_filename = filename.clone();
        let mut weights_filename = filename.clone();

        archi_filename.push_str(".arch");
        weights_filename.push_str(".wab");

        let  archi_file: File = File::open(archi_filename).expect("Error while creating the file: {archi_filename}");
        let  weights_file: File = File::open(weights_filename).expect("Error while creating the file: {weights_filename}");

        let mut archi_reader = BufReader::new(archi_file);
        let mut weights_reader = BufReader::new(weights_file);

        let mut buffer: String = String::new();

        // getting the number of layers 
        archi_reader.read_line(&mut buffer).expect("Failed to read the number of layers.");

        let nb_layers: usize = buffer.trim().parse::<usize>().expect("Cannot parse the supposed number of layers.");

        let mut weights: Vec<Matrix> = Vec::with_capacity(nb_layers - 1);
        let mut biases: Vec<Matrix> = Vec::with_capacity(nb_layers - 1);

        let mut values: Vec<Matrix> = Vec::with_capacity(nb_layers);
        let mut raw_values: Vec<Matrix> = Vec::with_capacity(nb_layers);

        let activations: Vec<DenseActivation>;
        let loss: DenseLosses;


        buffer = String::new();
        archi_reader.read_line(&mut buffer).expect("Failed to read the structure of each layers");
        // getting the structure of each layer
        let structures: Vec<usize> = buffer.trim().split(' ').map(|x| {x.parse::<usize>().expect("Cannot parse the structure of each layer.")}).collect();

        for i in 0..(structures.len() - 1) {
            values.push(Matrix::new(1,structures[i]));
            raw_values.push(Matrix::new(1,structures[i]).shuffle());

            weights.push(Matrix::new(structures[i], structures[i + 1]));
            biases.push(Matrix::new(1, structures[i + 1]));
        }
        values.push(Matrix::new(1,structures[structures.len() - 1]));
        raw_values.push(Matrix::new(1,structures[structures.len() - 1]));

        buffer = String::new();
        archi_reader.read_line(&mut buffer).expect("Failed to read the activation function of each layer.");
        activations = buffer.trim().split(' ').map(|x| {DenseActivation::from_str(x).expect("Cannot parse the activation function")}).collect();

        buffer = String::new();
        archi_reader.read_line(&mut buffer).expect("Failed to read the cost function of the model.");
        buffer = buffer.trim().to_string();
        loss = DenseLosses::from_str(&buffer).expect("Failed to parse the cost function of the model.");


        for l in 0..(nb_layers - 1) {

            for i in 0..weights[l].y_length {

                buffer = String::new();
                weights_reader.read_line(&mut buffer).expect("Failed to read the weight line.");
                let weights_values: Vec<f64> = buffer.trim().split(' ').map(|x| {x.parse::<f64>().expect("Cannot parse a weight value into a floating point.")}).collect();

                for j in 0..weights[l].x_length {
                    weights[l].set(i,j,weights_values[j]);
                }

                buffer = String::new();
                weights_reader.read_line(&mut buffer).expect("Failed to read the bias line.");
                biases[l].set(i,0, buffer.trim().parse::<f64>().expect("Cannot parse a bias value into a floating point."));
            }
        };

        DenseModel {
            nb_layers: nb_layers,
            loss: loss,
            activations: activations,
            weights: weights,
            biases: biases,
            raw_values: raw_values,
            values: values
        }
    } 
}