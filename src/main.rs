#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(warnings)]

mod activation_functions;
mod error_estimates;
mod models;
mod perceptrons;

fn main() {
    let inputs: Vec<Vec<f32>> = vec![
        // Negative class
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![2.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0],
        vec![1.5, 0.5, 0.0],
        vec![0.5, 1.5, 0.0],
        // Positive class
        vec![2.0, 1.0, 1.0],
        vec![1.0, 2.0, 1.0],
        vec![2.0, 2.0, 1.0],
        vec![3.0, 1.0, 1.0],
        vec![1.0, 3.0, 1.0],
        vec![3.0, 2.0, 1.0],
    ];

    let bias: f32 = 0.0;
    let mut weights: Vec<f32> = vec![0.0, 0.0];
    let eta: f32 = 0.1;
    let mut error: f32 = 0.0;
    let epochs = 6;

    let mut single_perceptron = models::singal_percetron_model::new(
        &inputs,
        2,
        activation_functions::ActivationFunctions::Step,
        epochs,
        eta,
    );

    single_perceptron.learn();
    let tst1: Vec<f32> = vec![0.0, 0.0];
    println!("OUTPUT1: {}", single_perceptron.predict(&tst1));

    let tst2: Vec<f32> = vec![1.0, 0.0];
    println!("OUTPUT2: {}", single_perceptron.predict(&tst2));
    single_perceptron.get_model_info();
}
