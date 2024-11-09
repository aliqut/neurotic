#![allow(dead_code)]

use ndarray::{Array1, Array2};

use super::optimiser::Optimiser;

/// RMSProp optimiser. Adjusts the learning rate based on a moving average of recent gradients.
pub struct RMSProp {
    /// Decay rate for the moving average.
    pub decay_rate: f32,

    /// Small epsilon value to prevent division by zero.
    pub epsilon: f32,

    /// Cache for accumulated squared weight gradients.
    pub weight_cache: Vec<Array2<f32>>,

    /// Cache for accumulated squared bias gradients.
    pub bias_cache: Vec<Array1<f32>>,

    weight_gradients: Vec<Array2<f32>>,
    bias_gradients: Vec<Array1<f32>>,
}

impl RMSProp {
    /// Creates a new `RMSProp` based on given inputs.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` -.
    /// * `decay_rate` -.
    /// * `epsilon` - .
    ///
    /// # Returns
    ///
    /// A new instance of `RMSProp`.

    pub fn new(layer_sizes: &[(usize, usize)], decay_rate: f32, epsilon: f32) -> Self {
        let weight_cache = layer_sizes
            .iter()
            .map(|(rows, columns)| Array2::zeros((*rows, *columns)))
            .collect();
        let bias_cache = layer_sizes
            .iter()
            .map(|(rows, _)| Array1::zeros(*rows))
            .collect();
        let weight_gradients = layer_sizes
            .iter()
            .map(|(rows, columns)| Array2::zeros((*rows, *columns)))
            .collect();
        let bias_gradients = layer_sizes
            .iter()
            .map(|(rows, _)| Array1::zeros(*rows))
            .collect();

        Self {
            decay_rate,
            epsilon,
            weight_cache,
            bias_cache,
            weight_gradients,
            bias_gradients,
        }
    }
}

impl Optimiser for RMSProp {
    fn update(
        &mut self,
        weight_gradients: &Array2<f32>,
        bias_gradients: &Array1<f32>,
        layer_index: usize,
    ) {
        self.weight_gradients[layer_index] = weight_gradients.clone();
        self.bias_gradients[layer_index] = bias_gradients.clone();

        self.weight_cache[layer_index] = self.decay_rate * &self.weight_cache[layer_index]
            + (1.0 - self.decay_rate) * weight_gradients.mapv(|g| g * g);
        self.bias_cache[layer_index] = self.decay_rate * &self.bias_cache[layer_index]
            + (1.0 - self.decay_rate) * bias_gradients.mapv(|g| g * g);
    }

    fn get_updated_biases(&self, layer_index: usize, learning_rate: f32) -> Array1<f32> {
        &self.bias_gradients[layer_index]
            * &self.bias_cache[layer_index]
                .mapv(|cache| -learning_rate / (cache.sqrt() + self.epsilon))
    }

    fn get_updated_weights(&self, layer_index: usize, learning_rate: f32) -> Array2<f32> {
        &self.weight_gradients[layer_index]
            * &self.weight_cache[layer_index]
                .mapv(|cache| -learning_rate / (cache.sqrt() + self.epsilon))
    }
}
