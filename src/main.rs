#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(warnings)]

use crate::{
    activation_functions::ActivationFunctions,
    error_estimates::errorTypes,
    models::{Ann, layer},
};

mod activation_functions;
mod error_estimates;
mod models;
mod perceptrons;

fn main() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0],
    ];
    let outputs: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];

    let mut layer1: models::layers::linearLayer =
        models::layers::linearLayer::new("linear layer 1".to_string(), 0, 2, 3);
    for input in &inputs {
        layer1.compute_linear_sum(input);
        layer1.print_output();
    }
}
