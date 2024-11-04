use rand::seq::SliceRandom;

use crate::core::NeuralNetwork;

use super::Batch;

#[derive(Clone, Debug)]
pub struct NetworkTrainer {
    network: NeuralNetwork,
    learning_rate: f32,
    batch_size: usize,
}

impl NetworkTrainer {
    pub fn new(network: NeuralNetwork, learning_rate: f32, batch_size: usize) -> Self {
        Self {
            network,
            learning_rate,
            batch_size,
        }
    }

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

    pub fn get_network(&mut self) -> &mut NeuralNetwork {
        &mut self.network
    }
}
