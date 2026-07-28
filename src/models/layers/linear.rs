use super::super::neuron;
use crate::error_estimates::errorTypes;
use crate::{
    activation_functions::ActivationFunctions, models::single_perceptron_model,
    perceptrons::single_perceptron,
};
use std::thread;
pub struct linearLayer {
    layerName: String,
    layerIndex: usize,
    outputLayer: Vec<f32>,
    totalNeurons: usize,
    neurons: Vec<neuron>,
    inputVector_length: usize,
}

impl linearLayer {
    pub fn new(
        layerName: String,
        layerIndex: usize,
        inputVector_length: usize,
        totalNeurons: usize,
    ) -> Self {
        let mut outputLayer: Vec<f32> = vec![0.0; totalNeurons];
        let mut neurons: Vec<neuron> = vec![];
        for i in (0..totalNeurons) {
            neurons.push(neuron::new(i, inputVector_length));
        }

        Self {
            layerName,
            layerIndex,
            outputLayer,
            totalNeurons,
            neurons,
            inputVector_length,
        }
    }
    pub fn compute_linear_sum(&mut self, inputVector: &Vec<f32>) {
        thread::scope(|s| {
            let mut handles = Vec::new();
            for neuron in &mut self.neurons {
                handles.push(s.spawn(|| (neuron.index, neuron.compute_weighted_sum(inputVector))));
            }
            for handle in handles {
                let (index, output) = handle.join().unwrap();
                self.outputLayer[index] = output;
            }
        })
    }
    pub fn print_output(&self) {
        println!("{:?}", self.outputLayer);
    }
}
