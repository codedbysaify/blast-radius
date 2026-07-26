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
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ];

    let mut layer1: models::layers::linearLayer<'_> = models::layers::linearLayer::new(
        "linear layer 1".to_string(),
        0,
        &inputs,
        true,
        Some(ActivationFunctions::Step),
        3,
        Some(errorTypes::Simple),
    );
    layer1.compute_weighted_sum();
}
