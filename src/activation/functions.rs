use serde::{Deserialize, Serialize};

/// Enum representing the different activation functions used for the neural network's layers.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ActivationFunction {
    /// Rectified Linear Unit (ReLU)
    ReLU,
    /// Leaky ReLU
    LeakyReLU,
    /// Sigmoid
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Identity, linear output
    Identity,
}

impl ActivationFunction {
    /// Applies the activation function from the enum ActivationFunction to an input
    ///
    /// Arguments:
    ///
    /// * `x` - input value
    ///
    /// # Returns
    ///
    /// The output of the activation function on input
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

    /// Applies the derivative of the activation function from the enum ActivationFunction to an input
    ///
    /// Arguments:
    ///
    /// * `x` - input value
    ///
    /// # Returns
    ///
    /// The output of the derivative of activation function on input
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
