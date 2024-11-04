use rand::seq::SliceRandom;

use crate::core::NeuralNetwork;

use super::Batch;

/// Manages the training of a `NeuralNetwork`
#[derive(Clone, Debug)]
pub struct NetworkTrainer {
    network: NeuralNetwork,
    learning_rate: f32,
    batch_size: usize,
}

impl NetworkTrainer {
    /// Construct a new `Network Trainer`.
    ///
    /// # Arguments
    ///
    /// * `network` - The `NeuralNetwork` to be trained.
    /// * `learning_rate` - Multiplier for the neurons' weight and bias updates.
    /// * `batch_size` - Number of samples per training batch to divide the input data into.
    ///
    /// # Returns
    ///
    /// A new instance of `NetworkTrainer`.
    pub fn new(network: NeuralNetwork, learning_rate: f32, batch_size: usize) -> Self {
        Self {
            network,
            learning_rate,
            batch_size,
        }
    }

    /// Train the network with specified input data and number of epochs.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Vector of tuples containing the input data and the expected outputs.
    /// * `epochs` - Number of training iterations.
    pub fn train(&mut self, mut training_data: Vec<(Vec<f32>, Vec<f32>)>, epochs: usize) {
        let mut rng = rand::thread_rng();

        for epoch in 0..epochs {
            // Shuffle training data
            training_data.shuffle(&mut rng);

            // Create batches
            let batches: Vec<_> = training_data
                .chunks(self.batch_size)
                .map(|chunk| {
                    let (inputs, targets): (Vec<_>, Vec<_>) = chunk.iter().cloned().unzip();
                    Batch::new(inputs, targets)
                })
                .collect();

            let total_loss: f32 = batches
                .iter()
                .map(|batch| self.network.train_batch(batch, self.learning_rate))
                .sum();

            let avg_loss = total_loss / batches.len() as f32;
            if epoch % 10 == 0 {
                println!(
                    "Epoch {}/{}: Average loss = {}",
                    epoch + 1,
                    epochs,
                    avg_loss
                );
            }
        }
    }

    /// Returns a mutable reference to the `NeuralNetwork` associated with the `NetworkTrainer`.
    pub fn get_network(&mut self) -> &mut NeuralNetwork {
        &mut self.network
    }
}
