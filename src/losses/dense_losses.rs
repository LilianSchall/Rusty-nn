use crate::maths::matrices::Matrix;
use crate::maths::INFINITY;

use std::process;


pub enum DenseLosses {
    NoLoss,
    CategoricalCrossEntropy,
    BinaryCrossEntropy,
    MeanSquaredError,
    CustomLoss
}

pub fn calculate_error(loss: DenseLosses, values: &Matrix, desired_output: &Matrix) -> f64 {

    match loss {
        DenseLosses::NoLoss | DenseLosses::CustomLoss => {
            println!("No Activation function, exiting..."); 
            process::exit(1);
        }
        DenseLosses::CategoricalCrossEntropy => categorical_cross_entropy(values, desired_output),
        DenseLosses::BinaryCrossEntropy => binary_cross_entropy(values, desired_output),
        DenseLosses::MeanSquaredError => mean_squared_error(values, desired_output)
    }
}

fn categorical_cross_entropy(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        if values.get(i,0) == 0.0 {
            
            if (desired_output.get(i,0) == 0.0) {
                continue;
            }
            else {
                sum += -INFINITY;
                return -sum;
            }
        }
        sum += desired_output.get(i,0) * values.get(i,0).ln();
    }
    -sum
}

fn binary_cross_entropy(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        sum += desired_output.get(i,0) * values.get(i,0).ln() + 
             (1.0 - desired_output.get(i,0)) * (1.0 - values.get(i,0)).ln();
    }

    -sum
}

fn mean_squared_error(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        sum += (desired_output.get(i,0) - values.get(i,0)).powi(2);
    }

    (1.0 / values.y_length as f64) * sum

}