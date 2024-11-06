use ndarray::{Array1, Array2, ArrayView1};
use rand::{distributions::Uniform, prelude::Distribution, thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::activation::ActivationFunction;

/// Represents a layer in the neural network. Each layer has a defined activation function, and two
/// arrays, one for weights and the other for biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weight matrix.
    pub weights: Array2<f32>,
    /// Bias matrix.
    pub biases: Array1<f32>,
    /// Activation function applied to the layer's outputs.
    pub activation_function: ActivationFunction,
}

impl Layer {
    /// Creates a new `Layer` with initialised weights and biases.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input neurons.
    /// * `output_size` - Number of output neurons.
    /// * `activation_function` - ActivationFunction enum variant to use for the layer's outputs.
    ///
    /// # Returns
    ///
    /// A new instance of `Layer` with random weights and zeroed biases.
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);

        let weight_scale = (2.0 / (input_size + output_size) as f32).sqrt();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            uniform.sample(&mut rng) * weight_scale
        });

        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            activation_function,
        }
    }

    /// Performs a forward pass on the layer with the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input array.
    ///
    /// # Returns
    ///
    /// A tuple containing the pre-activation (`z`) and the activation (`a`) outputs.
    pub fn forward(&self, input: &ArrayView1<f32>) -> (Array1<f32>, Array1<f32>) {
        let z = self.weights.dot(input) + &self.biases;
        let mut activation = z.clone();

        // If the input is large, use parallel mapping
        if z.len() > 1000 {
            activation.par_mapv_inplace(|x| self.activation_function.activate(x));
        } else {
            activation.mapv_inplace(|x| self.activation_function.activate(x));
        }

        (z, activation)
    }

    /// Performs a backward pass on the layer with the given input.
    ///
    /// # Arguments
    ///
    /// * `delta` - Error term for the current layer.
    /// * `prev_activation` - Previous layer's activation values.
    /// * `z` - Previous layer's pre-activation values.
    ///
    /// # Returns
    ///
    /// A tuple containing weight gradients, bias gradients, and propagated error term.
    pub fn backward(
        &self,
        delta: &Array1<f32>,
        prev_activation: &Array1<f32>,
        z: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        // Create array to store derivatives
        let mut z_derivative = z.clone();

        // Compute in parallel for large values
        if z.len() > 1000 {
            z_derivative.par_mapv_inplace(|x| self.activation_function.derivative(x));
        } else {
            z_derivative.mapv_inplace(|x| self.activation_function.derivative(x));
        }

        // Calculate loss gradient
        let delta = delta * z_derivative;

        // Calculate weight gradients
        let weight_gradients = delta.view().to_shape((delta.len(), 1)).unwrap().dot(
            &prev_activation
                .view()
                .to_shape((1, prev_activation.len()))
                .unwrap(),
        );

        let prev_delta = self.weights.t().dot(&delta);

        (weight_gradients, delta.clone(), prev_delta)
    }

    /// Updates the layer's parameters based on provided gradients.
    ///
    /// # Arguments
    ///
    /// * `weight_update` - Gradient for weights.
    /// * `bias_update` - Gradient for biases.
    /// * `learning_rate` - Multiplier for the weight and bias gradients to be applied.
    pub fn update_parameters(
        &mut self,
        weight_update: &Array2<f32>,
        bias_update: &Array1<f32>,
        learning_rate: f32,
    ) {
        self.weights -= &(learning_rate * weight_update);
        self.biases -= &(learning_rate * bias_update);
    }
}
