use crate::activation_functions::ActivationFunctions;
#[derive(Debug)]
pub struct singal_percetron_model<'a> {
    pub inputs: &'a Vec<Vec<f32>>,
    pub weights: &'a Vec<f32>,
    pub bias: f32,
    pub inputSize: i32,
    pub activate_function: ActivationFunctions,
    pub epochs: i32,
    pub eta: f32,
}

impl<'a> singal_percetron_model<'a> {
    pub fn new(
        inputs: &'a Vec<Vec<f32>>,
        weights: &'a Vec<f32>,
        bias: f32,
        inputSize: i32,
        activate_function: ActivationFunctions,
        epochs: i32,
        eta: f32,
    ) -> Self {
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
        println!("I am learning");
    }

    fn predict() {}
}
