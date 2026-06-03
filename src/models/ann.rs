use crate::activation_functions::ActivationFunctions;
use crate::error_estimates::errorTypes;
pub struct ann_model<'a> {
    pub layers: i32,
    pub nuerons: &'a Vec<i32>, //Tells the number of neuron in each layer, the last index tells about output layer neurons
    pub activation_function: &'a Vec<Vec<ActivationFunctions>>, // tells the activation function for each neuron
    pub eta: i32,
    pub loss_function: errorTypes,
    pub weight_initailization_method: String, // not implemented yet
    pub epochs: i32,
    pub batchSize: i32,               //default 1
    pub optimizer: String, // uses griadient loss to update the weights -> not implemented yet
    pub weight_update_method: String, // not implemented yet default backpropogation
}

pub struct Compiled_network {}

impl<'a> ann_model<'a> {
    pub fn new(
        layers: i32,
        nuerons: &'a Vec<i32>,
        activation_function: &'a Vec<Vec<ActivationFunctions>>,
        eta: i32,
        loss_function: errorTypes,
        weight_initailization_method: String,
        epochs: i32,
        batchSize: i32,
        optimizer: String,
        weight_update_method: String,
    ) -> Self {
        Self {
            layers,
            nuerons,
            activation_function,
            eta,
            loss_function,
            weight_initailization_method,
            epochs,
            batchSize,
            optimizer,
            weight_update_method,
        }
    }

    pub fn compile_network(&self) -> Compiled_network {
        Compiled_network {}
    }
}
