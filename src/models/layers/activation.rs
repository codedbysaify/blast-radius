use crate::activation_functions::{ActivationFunctions,step_activate};
use std::thread;

#[derive(Debug)]
pub struct Activation_layer<'a> {
    pub batchActivation: bool,
    pub activationFunction: Option<ActivationFunctions>,
    pub batch_activation_functions: Option<Vec<ActivationFunctions>>,
    pub inputVector: Option<&'a Vec<f32>>,
    pub outputVector: Vec<f32>,
}

impl<'a> Activation_layer<'a> {
    pub fn new(
        batchActivation: bool,
        activationFunction: Option<ActivationFunctions>,
        batch_activation_functions: Option<Vec<ActivationFunctions>>,
    ) -> Self {
        let mut outputVector: Vec<f32> = Vec::new();
        match batchActivation {
            true => {
                return Self {
                    batchActivation,
                    activationFunction,
                    batch_activation_functions: batch_activation_functions,
                    inputVector: None,
                    outputVector,
                };
            }
            false => {
                return Self {
                    batchActivation,
                    activationFunction,
                    batch_activation_functions: batch_activation_functions,
                    inputVector: None,
                    outputVector,
                };
            }
            _ => {
                return Self {
                    batchActivation: false,
                    activationFunction: Some(ActivationFunctions::Step),
                    batch_activation_functions: None,
                    inputVector: None,
                    outputVector,
                };
            }
        }
    }
    pub fn set_input_vector(&mut self, inputVector: &'a Vec<f32>) {
        self.inputVector = Some(inputVector);
    }
    pub fn apply_activation(&mut self) {
      
    }
}
