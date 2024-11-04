use serde::{Deserialize, Serialize};

/// Enum representing the different cost functions used to compute the cost of the output data with
/// respect to the expected target outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostFunction {
    /// Mean Squared Error
    MeanSquaredError,
    /// Cross Entropy
    CrossEntropy,
}

impl CostFunction {
    /// Calculates the cost for a given output and target.
    ///
    /// # Arguments
    ///
    /// * `output` - The `NeuralNetwork`s output neuron activations.
    /// * `target` - The `NeuralNetwork`s output neuron expected/target activations.
    pub fn calculate(&self, output: &[f32], target: &[f32]) -> f32 {
        match self {
            Self::MeanSquaredError => {
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&output, &target)| (output - target).powi(2) / 2.0)
                    .sum::<f32>()
                    / output.len() as f32
            }
            Self::CrossEntropy => -output
                .iter()
                .zip(target.iter())
                .map(|(&output, &target)| {
                    target * output.ln() + (1.0 - target) * (1.0 - output).ln()
                })
                .sum::<f32>(),
        }
    }

    /// Calculates the derivative of the cost function for a given output and target.
    ///
    /// # Arguments
    ///
    /// * `output` - The `NeuralNetwork`s output neuron activations.
    /// * `target` - The `NeuralNetwork`s output neuron expected/target activations.
    pub fn derivative(&self, output: f32, target: f32) -> f32 {
        match self {
            Self::MeanSquaredError => output - target,
            Self::CrossEntropy => (output - target) / (output * (1.0 - output)),
        }
    }
}
