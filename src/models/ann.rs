use super::single_perceptron_model::singal_percetron_model;

pub struct layer<'a>{
    pub position:i32, //position of the layer in the NN
    pub number_of_neurons: i32,//Total number of neurons 
    pub neuronsVector: Vec<singal_percetron_model<'a>>,


}

pub struct Ann{

}