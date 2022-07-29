use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::data::create_data::{Sample, load_data};
use crate::models::dense_model::DenseModel;
use crate::losses::dense_losses::calculate_error;

pub struct Session {
    pub dataset: Vec<Sample>,

    pub nb_epochs: usize,
    pub learning_rate: f64,

    pub loss_threshold: f64,
    pub stop_on_loss_threshold: bool,

}

impl Session {
    pub fn new(input_path: String, output_path: String,
        nb_epochs: usize, learning_rate: f64,
        loss_threshold: f64, stop_on_loss_threshold: bool) -> Session {
        
        let dataset: Vec<Sample> = load_data(&input_path, &output_path);

        Session {
            dataset: dataset,
            nb_epochs: nb_epochs,
            learning_rate: learning_rate,
            loss_threshold: loss_threshold,
            stop_on_loss_threshold: stop_on_loss_threshold,
        }
    }

    pub fn train(&mut self, model: &mut DenseModel) {

        let mut rng = thread_rng();
        
        for i in 0..self.nb_epochs {
            
            self.dataset.shuffle(&mut rng);
            let mut loss_buffer: f64 = 0.0;
            
            for j in 0..self.dataset.len() {

                model.feed_forward(&self.dataset[j].input);
                
                let error: f64 = calculate_error(&model.loss, &model.result(), &self.dataset[j].output);
                
                loss_buffer += error;

                let deltas = model.back_propagate(&self.dataset[j].output);
                model.update_weights(&deltas, self.learning_rate);

                if i == self.nb_epochs - 1{
                    println!("Sample:");
                    self.dataset[j].print_sample();
                    println!("guessed output:");
                    model.result().print();
                }
            }

            let avg_loss: f64 = loss_buffer / (self.dataset.len() as f64);
            if i % 1000 == 0 {
                println!("Epoch nb: {i} done: average loss = {avg_loss}");
            }
        }


    }
}