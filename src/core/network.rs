use ndarray::{Array1, Array2};
use std::{
    fs::File,
    io::{self, Read, Write},
};

use rmp_serde::{decode, encode};
use serde::{Deserialize, Serialize};

use crate::{
    activation::ActivationFunction,
    training::{Batch, CostFunction},
};

use super::layer::Layer;

/// The NeuralNetwork struct represents a multi-layered neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Vector containing each `Layer` in the network.
    pub layers: Vec<Layer>,
    /// The cost function used for computing cost when training the network.
    pub cost_function: CostFunction,
}

impl NeuralNetwork {
    /// Constructs a new `NeuralNetwork`
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Array representing the number of neurons in each layer.
    /// * `activation_functions` - Array representing the activation function used to compute the
    /// outputs of each layer.
    /// * `cost_function` - CostFunction enum for selecting the calculation used to compute the
    /// cost.
    ///
    /// # Returns
    ///
    /// A new instance of `NeuralNetwork`
    pub fn new(
        layer_sizes: &[usize],
        activation_functions: &[ActivationFunction],
        cost_function: CostFunction,
    ) -> Self {
        // Check if the amount of layer sizes matches the amount of activation functions
        assert!(layer_sizes.len() == activation_functions.len());

        let mut layers = Vec::new();

        for i in 1..layer_sizes.len() {
            layers.push(Layer::new(
                layer_sizes[i - 1],
                layer_sizes[i],
                activation_functions[i].clone(),
            ));
        }

        Self {
            layers,
            cost_function,
        }
    }

    /// Perform a forward propagation through the `NeuralNetwork` layers.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input data.
    ///
    /// # Returns
    ///
    /// A vector of tuples with pre-activation and activation values for each layer.
    pub fn forward(&self, input: &Array1<f32>) -> Vec<(Array1<f32>, Array1<f32>)> {
        let mut activations = vec![];
        let mut current_activation = input.clone();

        for layer in &self.layers {
            let (z, activation) = layer.forward(&current_activation);
            activations.push((z, activation.clone()));
            current_activation = activation;
        }

        activations
    }

    /// Train the network a batch of data.
    ///
    /// # Arguments
    ///
    /// * `batch` - A batch of data, containing input data and target outputs.
    /// * `learning_rate` - Multiplier for the neurons' weight and bias updates.
    ///
    /// # Returns
    ///
    /// The average loss over each batch.
    pub fn train_batch(&mut self, batch: &Batch, learning_rate: f32) -> f32 {
        let batch_size = batch.inputs.len() as f32;

        let mut total_loss = 0.0;

        let mut weight_gradients: Vec<Array2<f32>> = vec![];
        let mut bias_gradients: Vec<Array1<f32>> = vec![];

        // Initialise weight and bias gradient array vectors with zeros based on the layer's
        // dimensions
        for layer in &self.layers {
            weight_gradients.push(Array2::zeros(layer.weights.dim()));
            bias_gradients.push(Array1::zeros(layer.biases.dim()));
        }

        for (input, target) in batch.inputs.iter().zip(&batch.targets) {
            let input = Array1::from_vec(input.clone());
            let target = Array1::from_vec(target.clone());

            let layer_activations = self.forward(&input);

            let output = &layer_activations.last().unwrap().1;
            total_loss += self
                .cost_function
                .calculate(&output.to_vec(), &target.to_vec());

            let mut delta = (output - &target)
                * &output.map(|&x| {
                    self.layers
                        .last()
                        .unwrap()
                        .activation_function
                        .derivative(x)
                });

            for i in (0..self.layers.len()).rev() {
                let (weight_gradient, bias_gradient, new_delta) = if i > 0 {
                    self.layers[i].backward(
                        &delta,
                        &layer_activations[i - 1].1,
                        &layer_activations[i].0,
                    )
                } else {
                    self.layers[i].backward(&delta, &input, &layer_activations[i].0)
                };

                if weight_gradients.len() <= i {
                    weight_gradients.push(weight_gradient);
                    bias_gradients.push(bias_gradient);
                } else {
                    weight_gradients[i] += &weight_gradient;
                    bias_gradients[i] += &bias_gradient;
                }

                if i > 0 {
                    delta = new_delta;
                }
            }
        }

        for i in 0..self.layers.len() {
            self.layers[i].update_parameters(
                &(&weight_gradients[i] / batch_size),
                &(&bias_gradients[i] / batch_size),
                learning_rate,
            );
        }

        total_loss / batch_size
    }

    /// Saves the `NeuralNetwork` struct to a file.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the file where the network will be stored.
    ///
    /// # Returns
    ///
    /// Returns an io::Result.
    pub fn save(&self, filename: &str) -> io::Result<()> {
        // Serialize NeuralNetwork data using MessagePack
        let serialized = encode::to_vec(self).expect("Failed to serialize NeuralNetwork data");

        // Create file to store trained network
        let mut file = File::create(filename)?;

        // Write serialized data to the file
        file.write_all(&serialized)?;

        Ok(())
    }

    /// Load the `NeuralNetwork` struct from a file.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the file containing the `NeuralNetwork` data.
    ///
    /// # Returns
    ///
    /// Returns an io::Result containing the `NeuralNetwork` if loaded successfully.
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
