use rand::{Rng, RngExt};
#[derive(Debug, Clone)]
pub struct neuron {
    pub index: usize,
    pub inputVector_length: usize,
    pub output: f32,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl neuron {
    pub fn new(index: usize, inputVector_length: usize) -> Self {
        let mut rng = rand::rng();
        let mut weights: Vec<f32> = (0..inputVector_length)
            .map(|_| rng.random_range(-1.0..=1.0))
            .collect();

        Self {
            index,
            inputVector_length,
            output: 0.0,
            weights,
            bias: rng.random(),
        }
    }
    pub fn compute_weighted_sum(&mut self, inputVector: &Vec<f32>) -> f32 {
        let mut wx: f32 = 0.0;
        for i in (0..inputVector.len()) {
            wx += inputVector[i] * self.weights[i];
        }
        self.output = wx + self.bias;
        self.output
    }
}
