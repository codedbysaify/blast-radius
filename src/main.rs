
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

mod activation_functions;
mod models;
mod perceptrons;

fn main() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
    ];

    let mut bias: f32 = 0.0;
    let mut weights: Vec<f32> = vec![0.0, 0.0];
    let eta: f32 = 0.1;
    let mut error: f32 = 0.0;
    let epochs = 5;

    let mut single_perceptron: models::singal_percetron_model<'_> =
        models::singal_percetron_model::new(
            &inputs,
            &weights,
            bias,
            2,
            activation_functions::ActivationFunctions::Step,
            epochs,
            eta,
        );

    single_perceptron.learn();
}

// fn main() {
//     let inputs: Vec<(f32, f32, f32)> = vec![
//         (0.0, 0.0, 0.0),
//         (0.0, 1.0, 0.0),
//         (1.0, 1.0, 1.0),
//         (1.0, 0.0, 0.0),
//     ];

//     let mut bias: f32 = 0.0;
//     let mut weights: Vec<f32> = vec![0.0, 0.0];
//     let eta: f32 = 0.1;
//     let mut error: f32 = 0.0;
//     let epochs = 5;

//     for i in 0..=epochs {
//         println!("------EPOCh {}----", i);
//         for input in &inputs {
//             let input_vector: Vec<f32> = vec![input.0, input.1];
//             let net = perceptrons::perceptron(&input_vector, &weights, bias);
//             let output = activation_functions::step_activate(net);
//             println!("Output: {}", output);

//             // learning phase
//             error = input.2 - output;
//             println!("Total Error: {}", error);
//             weights[0] = weights[0] + (eta * error * (input.0));
//             weights[1] = weights[1] + (eta * error * (input.1));

//             println!("updated weights: {} , {}", weights[0], weights[1]);
//             bias = bias + eta * error;
//         }

//         println!("Updated weights : {}, {}", weights[0], weights[1]);
//         println!("Updated Bias: {}", bias);
//         println!("\n\n\n\n\n")
//     }
//     //prediction
//     println!("PREDICTION");
//     let net = perceptrons::perceptron(&[0.0, 1.0], &weights, bias);
//     let output = activation_functions::step_activate(net);
//     println!("final Output: {}", output);
// }
