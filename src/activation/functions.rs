use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Identity,
}

impl ActivationFunction {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::Identity => x,
        }
    }

    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Self::Sigmoid => {
                let s = self.activate(x);
                s * (1.0 - s)
            }

            Self::Tanh => 1.0 - x.tanh().powi(2),
            Self::Identity => 1.0,
        }
    }
}
