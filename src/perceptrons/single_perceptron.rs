pub fn compute_net(inputs: &[f32], weights: &[f32], bias: f32) -> f32 {
    if inputs.len() != weights.len() {
        panic!("Weights are not equal to total inputs");
    }
    let mut net: f32 = 0.0;
    for i in 0..=(inputs.len() - 1) {
        net = net + (inputs[i] * weights[i]);
    }
    net + bias
}
