use ndarray::{Array1, Array2};

use super::Optimiser;

/// Adam optimiser.
pub struct Adam {
    /// Exponential decay rate for first moment estimates.
    beta1: f32,
    /// Exponential decay rate for second moment estimates.
    beta2: f32,
    /// Small constant for numerical instability.
    epsilon: f32,
    /// Time step counter.
    t: usize,
    /// First moment of the weights.
    m_weights: Vec<Array2<f32>>,
    /// Second moment of the weights.
    v_weights: Vec<Array2<f32>>,
    /// First moment of the biases.
    m_biases: Vec<Array1<f32>>,
    /// Second moment of the biases.
    v_biases: Vec<Array1<f32>>,
    /// Store current gradients.
    weight_gradients: Vec<Array2<f32>>,
    /// Store current biases.
    bias_gradients: Vec<Array1<f32>>,
}

impl Adam {
    /// Creates a new `Adam` based on given inputs.
    ///
    /// # Arguments
    /// * `layer_sizes` - Dimensions of the layers.
    /// * `beta1` -  Exponential decay rate for first moment estimates.
    /// * `beta2` -  Exponential decay rate for second moment estimates.
    /// * `epsilon` - Small constant for numerical instability.
    pub fn new(layer_sizes: &[(usize, usize)], beta1: f32, beta2: f32, epsilon: f32) -> Self {
        let m_weights = layer_sizes
            .iter()
            .map(|(rows, columns)| Array2::zeros((*rows, *columns)))
            .collect();
        let v_weights = layer_sizes
            .iter()
            .map(|(rows, columns)| Array2::zeros((*rows, *columns)))
            .collect();
        let m_biases = layer_sizes
            .iter()
            .map(|(rows, _)| Array1::zeros(*rows))
            .collect();
        let v_biases = layer_sizes
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
            beta1,
            beta2,
            epsilon,
            t: 0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            weight_gradients,
            bias_gradients,
        }
    }

    fn get_bias_correction(&self) -> (f32, f32) {
        let correction1 = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let correction2 = 1.0 / (1.0 - self.beta2.powi(self.t as i32));
        (correction1, correction2)
    }
}

impl Optimiser for Adam {
    fn update(
        &mut self,
        weight_gradients: &Array2<f32>,
        bias_gradients: &Array1<f32>,
        layer_index: usize,
    ) {
        self.t += 1;

        self.weight_gradients[layer_index] = weight_gradients.clone();
        self.bias_gradients[layer_index] = bias_gradients.clone();

        self.m_weights[layer_index] =
            self.beta1 * &self.m_weights[layer_index] + (1.0 - self.beta1) * weight_gradients;
        self.m_biases[layer_index] =
            self.beta1 * &self.m_biases[layer_index] + (1.0 - self.beta1) * bias_gradients;

        self.v_weights[layer_index] = self.beta2 * &self.v_weights[layer_index]
            + (1.0 - self.beta2) * weight_gradients.mapv(|g| g * g);
        self.v_biases[layer_index] = self.beta2 * &self.v_biases[layer_index]
            + (1.0 - self.beta2) * bias_gradients.mapv(|g| g * g);
    }

    fn get_updated_weights(&self, layer_index: usize, learning_rate: f32) -> Array2<f32> {
        let (correction1, correction2) = self.get_bias_correction();

        let m_corrected = &self.m_weights[layer_index] * correction1;
        let v_corrected = &self.v_weights[layer_index] * correction2;

        -learning_rate * &(&m_corrected / &v_corrected.mapv(|v| (v + self.epsilon).sqrt()))
    }

    fn get_updated_biases(&self, layer_index: usize, learning_rate: f32) -> Array1<f32> {
        let (correction1, correction2) = self.get_bias_correction();

        let m_corrected = &self.m_biases[layer_index] * correction1;
        let v_corrected = &self.v_biases[layer_index] * correction2;

        -learning_rate * &(&m_corrected / &v_corrected.mapv(|v| (v + self.epsilon).sqrt()))
    }
}
