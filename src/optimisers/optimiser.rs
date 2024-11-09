use ndarray::{Array1, Array2};

/// Optimiser trait
pub trait Optimiser {
    /// Updates the gradients of weights and biases at the layer specified.
    ///
    /// # Arguments
    /// * `weight_gradients` - Gradients of weights.
    /// * `bias_gradients` - Gradients of biases.
    /// * `layer_index` - Index of the layer.
    fn update(
        &mut self,
        weight_gradients: &Array2<f32>,
        bias_gradients: &Array1<f32>,
        layer_index: usize,
    );

    /// Retrieves the updated weights for the layer specified after applying the optimizer.
    ///
    /// # Arguments
    /// * `layer_index` - Index of the layer.
    /// * `learning_rate` - Learning rate.
    ///
    /// # Returns
    /// Updated weight array.
    fn get_updated_weights(&self, layer_index: usize, learning_rate: f32) -> Array2<f32>;

    /// Retrieves the updated biases for the layer specified after applying the optimizer.
    ///
    /// # Arguments
    /// * `layer_index` - Index of the layer.
    /// * `learning_rate` - Learning rate.
    ///
    /// # Returns
    /// Updated bias array.
    fn get_updated_biases(&self, layer_index: usize, learning_rate: f32) -> Array1<f32>;
}
