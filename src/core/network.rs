use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::{
    fs::File,
    io::{self, Read, Write},
};

use rayon::iter::IntoParallelRefIterator;
use rmp_serde::{decode, encode};
use serde::{Deserialize, Serialize};

use crate::{
    activation::ActivationFunction,
    training::{Batch, CostFunction},
};

use super::layer::{Layer, LayerGradient};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub cost_function: CostFunction,
}

impl NeuralNetwork {
    pub fn new(
        layer_sizes: &[usize],
        activation_functions: &[ActivationFunction],
        cost_function: CostFunction,
    ) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() {
            let neuron_count = layer_sizes[i];
            let connection_count = {
                if i == 0 {
                    0
                } else {
                    layer_sizes[i - 1]
                }
            };
            layers.push(Layer::new(
                neuron_count,
                connection_count,
                activation_functions[i].clone(),
            ));
        }

        Self {
            layers,
            cost_function,
        }
    }

    pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut current_outputs = inputs.to_vec();

        for layer in self.layers.iter_mut().skip(1) {
            current_outputs = layer.forward(&current_outputs);
        }

        current_outputs
    }

    pub fn train_batch(&mut self, batch: &Batch<Vec<f32>>, learning_rate: f32) -> f32 {
        let batch_size = batch.inputs.len();

        // Struct to store layer neuron's activation values
        struct ForwardPassState {
            layer_activations: Vec<Vec<f32>>,
        }

        // Perform forward propogation for each input in the batch and store states
        let forward_states: Vec<ForwardPassState> = batch
            .inputs
            .iter()
            .map(|input| {
                let _output = self.forward(input);
                // Store activations from each layer
                let layer_activations: Vec<Vec<f32>> = self
                    .layers
                    .iter()
                    .map(|layer| layer.neurons.iter().map(|n| n.activation).collect())
                    .collect();

                ForwardPassState { layer_activations }
            })
            .collect();

        // Calculate total loss
        let total_loss: f32 = forward_states
            .par_iter()
            .zip(&batch.targets)
            .map(|(state, target)| {
                let output_layer_activations = state.layer_activations.last().unwrap();
                self.cost_function
                    .calculate(output_layer_activations, target)
            })
            .sum();

        // Calculate gradients
        let batch_gradients: Vec<Vec<LayerGradient>> = forward_states
            .par_iter()
            .zip(&batch.targets)
            .map(|(state, target)| {
                let mut layer_gradients: Vec<LayerGradient> = Vec::new();

                // Calculate output layer gradients
                let mut next_layer_delta = Vec::new();
                let output_layer_idx = self.layers.len() - 1;
                let output_activations = &state.layer_activations[output_layer_idx];

                for (i, activation) in output_activations.iter().enumerate() {
                    let cost_derivative = self.cost_function.derivative(*activation, target[i]);
                    let activation_derivative = self.layers[output_layer_idx]
                        .activation_function
                        .derivative(*activation);
                    next_layer_delta.push(cost_derivative * activation_derivative);
                }

                // Backpropagate hidden layers
                for layer_idx in (1..self.layers.len()).rev() {
                    let current_layer = &self.layers[layer_idx];
                    let prev_layer = &self.layers[layer_idx - 1];
                    let current_activations = &state.layer_activations[layer_idx];
                    let prev_activations = &state.layer_activations[layer_idx - 1];

                    let mut layer_gradient = LayerGradient {
                        weight_gradients: vec![
                            vec![0.0; prev_layer.neurons.len()];
                            current_layer.neurons.len()
                        ],
                        bias_gradients: vec![0.0; current_layer.neurons.len()],
                        delta: next_layer_delta.clone(),
                    };

                    // Calculate current layer gradients
                    for neuron_idx in 0..current_layer.neurons.len() {
                        let delta = layer_gradient.delta[neuron_idx];
                        layer_gradient.bias_gradients[neuron_idx] = delta;

                        for weight_idx in 0..prev_layer.neurons.len() {
                            let prev_activation = prev_activations[weight_idx];
                            layer_gradient.weight_gradients[neuron_idx][weight_idx] =
                                delta * prev_activation;
                        }
                    }

                    // If the layer isn't the first layer, calculate deltas for next iteration
                    if layer_idx > 1 {
                        let prev_layer = &self.layers[layer_idx - 1];
                        let mut new_deltas = vec![0.0; prev_layer.neurons.len()];

                        for prev_neuron_idx in 0..prev_layer.neurons.len() {
                            let mut error_sum = 0.0;

                            for (current_neuron_idx, neuron) in
                                current_layer.neurons.iter().enumerate()
                            {
                                error_sum += layer_gradient.delta[current_neuron_idx]
                                    * neuron.connections[prev_neuron_idx].weight;
                            }

                            new_deltas[prev_neuron_idx] = error_sum
                                * prev_layer
                                    .activation_function
                                    .derivative(prev_activations[prev_neuron_idx]);
                        }

                        next_layer_delta = new_deltas;
                    }

                    layer_gradients.push(layer_gradient);
                }

                layer_gradients.reverse();
                layer_gradients
            })
            .collect();

        // Apply the batch's accumulated gradients
        self.apply_gradients(batch_gradients, learning_rate, batch_size);

        total_loss / batch_size as f32
    }

    pub fn apply_gradients(
        &mut self,
        batch_gradients: Vec<Vec<LayerGradient>>,
        learning_rate: f32,
        batch_size: usize,
    ) {
        for layer_idx in 1..self.layers.len() {
            let layer = &mut self.layers[layer_idx];

            let layer_gradients: Vec<&LayerGradient> = batch_gradients
                .iter()
                .filter_map(|gradients| gradients.get(layer_idx - 1))
                .collect();

            layer.apply_gradients(&layer_gradients, learning_rate, batch_size as f32);
        }
    }

    pub fn save(&self, filename: &str) -> io::Result<()> {
        // Serialize NeuralNetwork data using MessagePack
        let serialized = encode::to_vec(self).expect("Failed to serialize NeuralNetwork data");

        // Create file to store trained network
        let mut file = File::create(filename)?;

        // Write serialized data to the file
        file.write_all(&serialized)?;

        Ok(())
    }

    pub fn load(filename: &str) -> io::Result<Self> {
        // Open file at "filename" path
        let mut file = File::open(filename)?;

        // Create vector to store file contents
        let mut contents = Vec::new();

        // Read file's contents to contents vector
        file.read_to_end(&mut contents)?;

        // Deserialize data into NeuralNetwork struct
        let network: NeuralNetwork =
            decode::from_slice(&contents).expect("Failed to deserialize NeuralNetwork data");

        Ok(network)
    }
}
