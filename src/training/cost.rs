use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostFunction {
    MeanSquaredError,
    CrossEntropy,
}

impl CostFunction {
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

    pub fn derivative(&self, output: f32, target: f32) -> f32 {
        match self {
            Self::MeanSquaredError => output - target,
            Self::CrossEntropy => (output - target) / (output * (1.0 - output)),
        }
    }
}
