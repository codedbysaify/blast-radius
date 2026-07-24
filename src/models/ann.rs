use super::single_perceptron_model::singal_percetron_model;
use crate::activation_functions::ActivationFunctions;
use crate::error_estimates::errorTypes;

#[derive(Debug)]
pub struct layer<'a> {
    pub position: i32,          //position of the layer in the NN
    pub number_of_neurons: i32, //Total number of neurons
    pub neuronsVector: &'a Vec<singal_percetron_model<'a>>,
    pub layerOutput: Vec<f32>,
    pub ActivationFunction: ActivationFunctions,
}
#[derive(Debug)]
pub struct Ann<'a> {
    pub totalLayers: i32,
    pub errorEstimate: errorTypes,
    pub compiledNetwork: &'a Vec<layer<'a>>,
    pub networkOutput: Vec<f32>,
}

impl<'a> Ann<'a> {
    pub fn new(
        totalLayers: i32,
        errorEstimate: errorTypes,
        compiledNetwork: &'a Vec<layer<'a>>,
        networkOutput: Vec<f32>,
    ) -> Self {
        Self {
            totalLayers,
            errorEstimate,
            compiledNetwork,
            networkOutput,
        }
    }

    pub fn getNetworkInfo(&self) {
        for layer in self.compiledNetwork {
            for nueron in layer.neuronsVector {
                println!("{:?}", nueron.get_info());
            }
            println!("----");
        }
    }
}
