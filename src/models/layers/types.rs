use super::activation::Activation_layer;
use super::linear::linearLayer;

#[derive(Debug)]
pub enum layerType {
    INPUT,
    LINEAR,
    ACTIVATION,
}

#[derive(Debug)]
pub enum layer_objects {
    LINEAR(linearLayer),
    ACTIVATION(Activation_layer),
}
