use crate::activation_functions::{ActivationFunctions, step_activate};
use std::thread;

#[derive(Debug)]
pub struct Activation_layer {
    pub layerName: String,
    pub batchActivation: bool,
    pub activationFunction: Option<ActivationFunctions>,
    pub batch_activation_functions: Option<Vec<ActivationFunctions>>,
    pub inputVector: Option<Vec<f32>>,
    pub outputVector: Vec<f32>,
}

impl Activation_layer {
    pub fn new(
        batchActivation: bool,
        activationFunction: Option<ActivationFunctions>,
        batch_activation_functions: Option<Vec<ActivationFunctions>>,
        layerName: String,
    ) -> Self {
        let mut outputVector: Vec<f32> = Vec::new();
        match batchActivation {
            true => {
                return Self {
                    layerName,
                    batchActivation,
                    activationFunction,
                    batch_activation_functions: batch_activation_functions,
                    inputVector: None,
                    outputVector,
                };
            }
            false => {
                return Self {
                    layerName,
                    batchActivation,
                    activationFunction,
                    batch_activation_functions: batch_activation_functions,
                    inputVector: None,
                    outputVector,
                };
            }
            _ => {
                return Self {
                    layerName,
                    batchActivation: false,
                    activationFunction: Some(ActivationFunctions::Step),
                    batch_activation_functions: None,
                    inputVector: None,
                    outputVector,
                };
            }
        }
    }
    pub fn set_input_vector(&mut self, inputVector: Vec<f32>) {
        self.outputVector.resize(inputVector.len(), 0.0);
        self.inputVector = Some(inputVector);
    }
    pub fn apply_activation(&mut self) {
        match self.inputVector {
            Some(_) => {}
            None => {
                panic!("Input vector is not defined please define it.")
            }
        }

        match self.batchActivation {
            true => {
                thread::scope(|s| {
                    let mut handles = Vec::new();
                    let inputVector = self
                        .inputVector
                        .as_ref()
                        .expect("Input vector cant be None");
                    let activationFunction = self.activationFunction.unwrap();
                    for i in 0..inputVector.len() {
                        handles.push(s.spawn(move || match activationFunction {
                            ActivationFunctions::Step => (i, step_activate(inputVector[i])),
                            _ => (i, step_activate(inputVector[i])),
                        }));
                    }
                    for handle in handles {
                        let (index, output) = handle.join().unwrap();
                        self.outputVector[index] = output;
                    }
                });
            }
            false => {}
            _ => {}
        }
    }
    pub fn get_output_vector(&mut self, inputVector: Vec<f32>) -> &Vec<f32> {
        self.set_input_vector(inputVector);
        self.apply_activation();
        &self.outputVector
    }
}
