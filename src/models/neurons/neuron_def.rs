use rand::{Rng, RngExt};
#[derive(Debug, Clone)]
pub struct neuron<'a> {
    pub index: usize,
    pub inputVector: &'a Vec<f32>,
    pub output: f32,
    pub weights: Vec<f32>,
    pub bias: f32,
}

impl<'a> neuron<'a> {
    pub fn new(index: usize, inputVector: &'a Vec<f32>) -> Self {
        let mut rng = rand::rng();
        let mut weights: Vec<f32> = (0..inputVector.len())
            .map(|_| rng.random_range(-1.0..=1.0))
            .collect();

        Self {
            index,
            inputVector,
            output: 0.0,
            weights,
            bias: rng.random(),
        }
    }
    pub fn compute_weighted_sum(&mut self) -> f32 {
        let mut wx: f32 = 0.0;
        for i in (0..self.inputVector.len()) {
            wx += self.inputVector[i] * self.weights[i];
        }
        self.output = wx + self.bias;
        self.output
    }
}
