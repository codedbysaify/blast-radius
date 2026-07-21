#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(warnings)]

use crate::error_estimates::errorTypes;

mod activation_functions;
mod error_estimates;
mod models;
mod perceptrons;

fn main() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ];

    let mut bias: f32 = 0.0;
    let mut weights: Vec<f32> = vec![0.0, 0.0];
    let eta: f32 = 0.1;
    let mut error: f32 = 0.0;
    let epochs = 7;

    let mut single_perceptron = models::singal_percetron_model::new(
        &inputs,
        2,
        activation_functions::ActivationFunctions::Step,
        eta,
        errorTypes::Simple,
    );

    single_perceptron.learn(epochs);
    let tst1: Vec<f32> = vec![0.0, 0.0];
    println!("OUTPUT1: {}", single_perceptron.predict(&tst1));

    let tst2: Vec<f32> = vec![1.0, 1.0];
    println!("OUTPUT2: {}", single_perceptron.predict(&tst2));
    single_perceptron.get_model_info();
}
