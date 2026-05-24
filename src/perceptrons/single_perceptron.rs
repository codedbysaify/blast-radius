pub fn perceptron(inputs: &[f32], weights: &[f32], bias: f32) -> f32 {
    println!(
        "Total inputs recieved {}: {},{}",
        inputs.len(),
        inputs[0],
        inputs[1]
    );
    println!("Total Weights recieved {}", weights.len());
    if inputs.len() != weights.len() {
        panic!("Weights are not equal to total inputs");
    }
    let mut net: f32 = 0.0;
    for i in 0..=(inputs.len() - 1) {
        net = net + (inputs[i] * weights[i]);
    }
    net + bias
}
