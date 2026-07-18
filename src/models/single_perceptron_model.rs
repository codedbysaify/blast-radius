use crate::activation_functions::{self, ActivationFunctions};
use crate::perceptrons::compute_net;
#[derive(Debug)]
pub struct singal_percetron_model<'a> {
    pub inputs: &'a Vec<Vec<f32>>,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub inputSize: usize,
    pub activate_function: ActivationFunctions,
    pub epochs: i32,
    pub eta: f32,
}

impl<'a> singal_percetron_model<'a> {
    pub fn new(
        inputs: &'a Vec<Vec<f32>>,
        inputSize: usize,
        activate_function: ActivationFunctions,
        epochs: i32,
        eta: f32,
    ) -> Self {
        let weights: Vec<f32> = vec![0.0; inputSize];
        let bias = 1.0;
        Self {
            inputs,
            weights,
            bias,
            inputSize,
            activate_function,
            epochs,
            eta,
        }
    }

    pub fn learn(&mut self) {
        for epoch in (0..self.epochs) {
            println!("Epoch: {}", epoch);
            for instance in self.inputs {
                let input = &instance[0..self.inputSize];
                let result = instance[self.inputSize];
                let output = self.predict(&input);

                //updating weights and bias
                let error: f32 = result - output;
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
