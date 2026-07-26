use super::super::perceptron_model;
use crate::error_estimates::errorTypes;
use crate::{
    activation_functions::ActivationFunctions, models::single_perceptron_model,
    perceptrons::single_perceptron,
};
pub struct linearLayer<'a> {
    layerName: String,
    layerIndex: usize,
    inputLayer: &'a Vec<Vec<f32>>,
    outputLayer: Vec<f32>,
    BatchActivation: bool, //if True all the neurons of the layer have same activation function
    activation_function: Option<ActivationFunctions>,
    totalNeurons: usize,
    neurons: Vec<perceptron_model<'a>>,
    errorEstimatingFunction: Option<errorTypes>,
}

impl<'a> linearLayer<'a> {
    pub fn new(
        layerName: String,
        layerIndex: usize,
        inputLayer: &'a Vec<Vec<f32>>,
        BatchActivation: bool,
        activation_function: Option<ActivationFunctions>,
        totalNeurons: usize,
        errorEstimatingFunction: Option<errorTypes>,
    ) -> Self {
        let mut outputLayer: Vec<f32> = vec![0.0; totalNeurons];
        let mut perceptron: perceptron_model<'a> = perceptron_model::new(
            inputLayer,
            2,
            activation_function.expect("Activation function is required in the perceptron"),
            0.1,
            errorEstimatingFunction.expect("Error estimating function type is required"),
        );
        let mut neurons: Vec<perceptron_model> = vec![perceptron; totalNeurons];

        Self {
            layerName,
            layerIndex,
            inputLayer,
            outputLayer,
            BatchActivation,
            activation_function,
            totalNeurons,
            neurons,
            errorEstimatingFunction,
        }
    }
    pub fn compute_weighted_sum(&mut self) {
        for neuron in &mut self.neurons {
            neuron.learn(6);
            neuron.get_info();
            println!("_____");
        }
    }
}
