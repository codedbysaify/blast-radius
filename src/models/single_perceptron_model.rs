use crate::activation_functions::{self, ActivationFunctions};
use crate::error_estimates::{errorTypes, simple_error};
use crate::perceptrons::compute_net;
#[derive(Debug)]
pub struct singal_percetron_model<'a> {
    pub inputs: &'a Vec<Vec<f32>>,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub inputSize: usize,
    pub activate_function: ActivationFunctions,
    pub eta: f32,
    pub errorType: errorTypes,
}

impl<'a> singal_percetron_model<'a> {
    pub fn new(
        inputs: &'a Vec<Vec<f32>>,
        inputSize: usize,
        activate_function: ActivationFunctions,
        eta: f32,
        errorType: errorTypes,
    ) -> Self {
        let weights: Vec<f32> = vec![0.0; inputSize];
        let bias = 1.0;
        Self {
            inputs,
            weights,
            bias,
            inputSize,
            activate_function,
            eta,
            errorType,
        }
    }

    pub fn learn(&mut self, epochs: i32) {
        for epoch in (0..epochs) {
            println!("Epoch: {}", epoch);
            for instance in self.inputs {
                let input = &instance[0..self.inputSize];
                let actual_output = instance[self.inputSize];
                let predicted_output = self.predict(&input);

                //updating weights and bias
                let error: f32 = match self.errorType {
                    errorTypes::Simple => simple_error(actual_output, predicted_output),
                    _ => actual_output - predicted_output,
                };
                for i in (0..self.inputSize) {
                    self.weights[i] += error * self.eta * input[i];
                }

                self.bias += error * self.eta;
            }
        }
    }
    pub fn predict(&self, inputs: &[f32]) -> f32 {
        let computed_net: f32 = compute_net(inputs, &self.weights, self.bias);
        activation_functions::step_activate(computed_net)
    }

    pub fn get_model_info(&self) {
        println!("{:?}", self);
    }
}
