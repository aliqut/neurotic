use serde::{Deserialize, Serialize};

use crate::activation::ActivationFunction;

use super::{neuron::Neuron, Connection};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation_function: ActivationFunction,
    pub activation_buffer: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerGradient {
    pub weight_gradients: Vec<Vec<f32>>,
    pub bias_gradients: Vec<f32>,
    pub delta: Vec<f32>,
}

impl Layer {
    pub fn new(
        neuron_count: usize,
        connection_count: usize,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut neurons = Vec::with_capacity(neuron_count);

        for _ in 0..neuron_count {
            let mut neuron = Neuron::new();
            if connection_count > 0 {
                for _ in 0..connection_count {
                    neuron.connections.push(Connection::new(connection_count));
                }
            }
            neurons.push(neuron);
        }

        Self {
            neurons,
            activation_function,
            activation_buffer: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: &[f32]) -> &[f32] {
        self.activation_buffer.clear();
        self.activation_buffer
            .extend(self.neurons.iter_mut().map(|neuron| {
                let weighted_sum: f32 = neuron
                    .connections
                    .iter()
                    .zip(inputs)
                    .map(|(connection, &input)| connection.weight * input)
                    .sum::<f32>()
                    + neuron.bias;

                neuron.activation = self.activation_function.activate(weighted_sum);
                neuron.activation
            }));
        &self.activation_buffer
    }

    pub fn apply_gradients(
        &mut self,
        gradients: &Vec<&LayerGradient>,
        learning_rate: f32,
        batch_size: f32,
    ) {
        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            let avg_bias_gradient: f32 = gradients
                .iter()
                .map(|gradient| gradient.bias_gradients[neuron_idx])
                .sum::<f32>()
                / batch_size;

            neuron.bias -= learning_rate * avg_bias_gradient;

            for (weight_idx, connection) in neuron.connections.iter_mut().enumerate() {
                let avg_weight_gradient: f32 = gradients
                    .iter()
                    .map(|gradient| gradient.weight_gradients[neuron_idx][weight_idx])
                    .sum::<f32>()
                    / batch_size;

                connection.weight -= learning_rate * avg_weight_gradient;
            }
        }
    }
}
