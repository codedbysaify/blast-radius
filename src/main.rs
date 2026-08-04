#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(warnings)]

use crate::models::layers::types::layer_objects;
use crate::{activation_functions::ActivationFunctions, error_estimates::errorTypes, models::Ann};

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

    let mut layer1: layer_objects = layer_objects::LINEAR(models::layers::linearLayer::new(
        "linear layer 1".to_string(),
        0,
        2,
        3,
    ));

    let mut layer2: layer_objects =
        layer_objects::ACTIVATION(models::layers::Activation_layer::new(
            true,
            Some(ActivationFunctions::Step),
            None,
            "layer2".to_string(),
        ));

    let mut annNetwork = Ann::new();
    annNetwork.addLayer(layer1);
    annNetwork.addLayer(layer2);

    for input in inputs {
        annNetwork.forwardPass(input);
    }
}
//  for input in &inputs {
//         layer1.compute_linear_sum(input);
//         layer1.print_output();
//         layer2.set_input_vector(layer1.get_output_vector().clone());
//         println!("Activation Layer Output");
//         layer2.apply_activation();
//         layer2.print_output();
//         println!("_____");
//     }
