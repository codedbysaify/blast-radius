use std::vec;

use super::layers::types::layer_objects;
use super::single_perceptron_model::perceptron_model;
use crate::activation_functions::ActivationFunctions;
use crate::error_estimates::errorTypes;
use crate::models::layers::{Activation_layer, linear};

#[derive(Debug)]
pub struct Ann {
    pub totalLayers: usize,
    pub network: Vec<layer_objects>,
    pub epochs: usize,
    pub eta: f32,
}

impl Ann {
    pub fn new() -> Self {
        Self {
            totalLayers: 0,
            network: Vec::new(),
            epochs: 0,
            eta: 0.0,
        }
    }

    pub fn addLayer(&mut self, layer: layer_objects) {
        self.network.push(layer);
        self.totalLayers += 1;
    }

    pub fn forwardPass(&mut self, inputVector: Vec<f32>) {
        let mut seedLayer: &Vec<f32> = &inputVector;
        for layer in &mut self.network {
            match layer {
                layer_objects::LINEAR(linearLayer) => {
                    seedLayer = linearLayer.get_output_vector(&seedLayer);
                }
                layer_objects::ACTIVATION(activationLayer) => {
                    seedLayer = activationLayer.get_output_vector(seedLayer.clone());
                }
            }
        }
        println!("Final Output: ");
        println!("{:?}", seedLayer);
    }
}
