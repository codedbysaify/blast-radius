use crate::activation_functions::{self, ActivationFunctions, step_activate};
use crate::error_estimates::{errorTypes, simple_error};
use crate::perceptrons::{perceptron, single_perceptron};
#[derive(Debug)]
pub struct singal_percetron_model<'a> {
    pub inputs: &'a Vec<Vec<f32>>,
    pub weights: &'a mut Vec<f32>,
    pub bias: f32,
    pub inputSize: usize,
    pub activate_function: ActivationFunctions,
    pub epochs: i32,
    pub eta: f32,
    pub error_type: errorTypes,
}

impl<'a> singal_percetron_model<'a> {
    pub fn new(
        inputs: &'a Vec<Vec<f32>>,
        weights: &'a mut Vec<f32>,
        bias: f32,
        inputSize: usize,
        activate_function: ActivationFunctions,
        epochs: i32,
        eta: f32,
        error_type: errorTypes,
    ) -> Self {
        Self {
            inputs,
            weights,
            bias,
            inputSize,
            activate_function,
            epochs,
            eta,
            error_type,
        }
    }

    pub fn learn(&mut self) {
        for i in 1..=self.epochs {
            println!(
                "
            ----------------------------\n
            EPOCh : {}
            ----------------------------\n
            ",
                i
            );

            for input in self.inputs {
                let inputVector = &input[0..self.inputSize];
                let net: f32 = perceptron(&inputVector, self.weights, self.bias);
                let output = match self.activate_function {
                    ActivationFunctions::Step => activation_functions::step_activate(net),

                    _ => activation_functions::step_activate(net),
                };
                let error: f32 = match self.error_type {
                    errorTypes::Simple => simple_error(input[self.inputSize], output),
                    _ => simple_error(input[self.inputSize], output),
                };

                for i in 0..self.weights.len() {
                    self.weights[i] = self.weights[i] + self.eta * error * inputVector[i];
                }
                self.bias = self.bias + self.eta * error;
            }
        }
    }

    pub fn print_updated_parameters(&self) {
        println!("\n┌──────────────────────────────────────────────┐");
        println!("│            UPDATED PARAMETERS               │");
        println!("└──────────────────────────────────────────────┘");

        println!("\n📌 WEIGHTS:");
        println!("{:#?}", self.weights);

        println!("\n📌 BIAS:");
        println!("{:.6}", self.bias);

        println!("\n──────────────────────────────────────────────");
    }

    pub fn predict(&self, input: &Vec<f32>) -> f32 {
        let net = perceptron(input, self.weights, self.bias);
        let output = match self.activate_function {
            ActivationFunctions::Step => step_activate(net),
            _ => step_activate(net),
        };
        output
    }
}
