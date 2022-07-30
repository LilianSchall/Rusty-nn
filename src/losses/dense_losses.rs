use crate::maths::matrices::Matrix;
use crate::maths::INFINITY;

use std::process;
use std::str::FromStr;


#[derive(strum_macros::Display)]
pub enum DenseLosses {
    NoLoss,
    CategoricalCrossEntropy,
    BinaryCrossEntropy,
    MeanSquaredError,
    CustomLoss
}

impl FromStr for DenseLosses {
    type Err = ();

    fn from_str(input: &str) -> Result<DenseLosses, Self::Err> {
        match input {
            "NoLoss"                    => Ok(DenseLosses::NoLoss),
            "CategoricalCrossEntropy"   => Ok(DenseLosses::CategoricalCrossEntropy),
            "BinaryCrossEntropy"        => Ok(DenseLosses::BinaryCrossEntropy),
            "MeanSquaredError"          => Ok(DenseLosses::MeanSquaredError),
            "CustomLoss"                => Ok(DenseLosses::CustomLoss),
            _ => Err(())
        }
    }
}

pub fn calculate_error(loss: &DenseLosses, values: &Matrix, desired_output: &Matrix) -> f64 {

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

pub fn derivative_error(loss: &DenseLosses, nb_values: usize, single_guess: f64, single_desired: f64) -> f64 {

    match loss {
        DenseLosses::NoLoss | DenseLosses::CustomLoss => {
            println!("No Activation function, exiting..."); 
            process::exit(1);
        }
        DenseLosses::CategoricalCrossEntropy =>
             d_categorical_cross_entropy(single_guess, single_desired),
        DenseLosses::BinaryCrossEntropy =>
             d_binary_cross_entropy(single_guess, single_desired),
        DenseLosses::MeanSquaredError =>
             d_mean_squared_error(nb_values, single_guess, single_desired)
    }
}



fn d_categorical_cross_entropy(single_guess: f64, single_desired: f64) -> f64 {

    - (single_desired / single_guess)

}

fn categorical_cross_entropy(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        if values.get(i,0) == 0.0 {
            
            if desired_output.get(i,0) == 0.0 {
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

fn d_binary_cross_entropy(single_guess: f64, single_desired: f64) -> f64 {

    - (single_desired / single_guess) + (1.0 - single_desired) / (1.0 - single_guess)

}

fn binary_cross_entropy(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        sum += desired_output.get(i,0) * values.get(i,0).ln() + 
             (1.0 - desired_output.get(i,0)) * (1.0 - values.get(i,0)).ln();
    }

    -sum
}

fn d_mean_squared_error(nb_values: usize, single_guess: f64, single_desired: f64) -> f64 {

    - (2.0 / (nb_values as f64)) * (single_desired - single_guess)

}

fn mean_squared_error(values: &Matrix, desired_output: &Matrix) -> f64 {

    let mut sum: f64 = 0.0;

    for i in 0..values.y_length {
        sum += (desired_output.get(i,0) - values.get(i,0)).powi(2);
    }

    (1.0 / values.y_length as f64) * sum 

}