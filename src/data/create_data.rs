use std::fs::File;
use std::io::{BufReader, BufRead};
use std::{process, usize};


pub struct Sample {
    pub input: Vec<f64>,
    pub output: Vec<f64>
}

impl Sample {

    pub fn new(input: Vec<f64>, output: Vec<f64>) -> Sample{
        Sample {
            input: input,
            output: output
        }
    }

    pub fn generate_sample_vec(input_vec: &mut Vec<Vec<f64>>, output_vec: &mut Vec<Vec<f64>>) -> Vec<Sample> {

        let mut sample_vec = Vec::with_capacity(input_vec.len());

        for _ in 0..input_vec.len() {
            sample_vec.push(Sample::new(input_vec.pop().unwrap(), output_vec.pop().unwrap()));
        }

        sample_vec
    }

    pub fn print_sample(&self) {
        println!("---Input---");
        for i in 0..self.input.len() {
            let x = self.input[i];
            print!("{x}, ");
        }
        println!("");

        println!("---Output---");
        for i in 0..self.output.len() {
            let x = self.output[i];
            print!("{x}, ");
        }
        println!("");
    }
}


pub fn load_data(input_path: &String, output_path: &String) -> Vec<Sample>{
    
    let (mut input, x_in, y_in, z_in) = generate_data_vec(input_path);
    let (mut output,x_out,y_out,z_out) = generate_data_vec(output_path);

    assert_eq!(input.len(), output.len());

    Sample::generate_sample_vec(&mut input, &mut output)
}


fn generate_data_vec(filepath: &String) -> (Vec<Vec<f64>>, usize, usize, usize){

    let file: File =  File::open(filepath).expect("File {filepath} not found");

    let mut reader = BufReader::new(file);

    let mut options_str: String = String::new();
    reader.read_line(&mut options_str).expect("An error occured while reading the first line of the file.");
    let options = parse_option(options_str, &filepath, &"An error occured while parsing options.".to_string());

    let nb_lines: usize = options[0];
    let x: usize = options[1];
    let y: usize = options[2];
    let z: usize = options[3];

    let mut data_vec: Vec<Vec<f64>> = Vec::with_capacity(nb_lines);

    read_data(&filepath, &mut reader, &mut data_vec, &"An error occured while reading the data.".to_string());

    (data_vec,x,y,z)
}

fn parse_option(options_str: String, filepath: &String, error_code: &String) -> Vec<usize> {
    let iterator = options_str.trim().split(' ').map(|x| {
        let this = x.parse::<usize>();
        match this {
            Ok(t) => t,
            Err(_) => { unwrap_failed(&error_code, &filepath)},
        }
    } as usize);

    Vec::from_iter(iterator)
}

fn read_data(filepath: &String, reader: &mut BufReader<File>, vec_to_fill: &mut Vec<Vec<f64>>, error_code: &String) {

    for line in reader.lines() {

        let content = Vec::from_iter(line
            .expect(&error_code)
            .trim()
            .split(' ')
            .map(|x| {
                let this = x.parse::<f64>();
                match this {
                    Ok(t) => t,
                    Err(e) => {
                        println!("{e}");
                        unwrap_failed(&error_code, &filepath) as f64
                    },
                }
            } as f64));
        
            vec_to_fill.push(content);
    }
} 

fn unwrap_failed(error_msg: &String, filepath: &String) -> usize {
    println!("{error_msg}");
    println!("This error has occured while reading the file: {filepath}");
    process::exit(1);
}