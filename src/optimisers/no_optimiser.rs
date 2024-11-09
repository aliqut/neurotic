use ndarray::{Array1, Array2};

use super::Optimiser;

/// A no-operation optimiser in cases where no optimiser is required. Satisfies the `Optimiser`
/// trait.
pub struct NoOptimiser {}

impl Optimiser for NoOptimiser {
    fn update(
        &mut self,
        _weight_gradients: &ndarray::Array2<f32>,
        _bias_gradients: &ndarray::Array1<f32>,
        _layer_index: usize,
    ) {
    }
    fn get_updated_biases(&self, _layer_index: usize, _learning_rate: f32) -> ndarray::Array1<f32> {
        Array1::zeros(0)
    }
    fn get_updated_weights(
        &self,
        _layer_index: usize,
        _learning_rate: f32,
    ) -> ndarray::Array2<f32> {
        Array2::zeros((0, 0))
    }
}
