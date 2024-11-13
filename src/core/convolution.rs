use ndarray::{Array1, Array3, Array4, ArrayView3, Axis};
use rand::{distributions::Uniform, prelude::Distribution};
use serde::{Deserialize, Serialize};

use crate::activation::ActivationFunction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvLayer {
    pub filters: Array4<f32>,
    pub biases: Array1<f32>,
    pub stride: usize,
    pub padding: usize,
    pub activation_function: ActivationFunction,
}

impl ConvLayer {
    pub fn new(
        input_size: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation_function: ActivationFunction,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);

        let weight_scale = (2.0 / (input_size * kernel_size * kernel_size) as f32).sqrt();

        let filters =
            Array4::from_shape_fn((num_filters, input_size, kernel_size, kernel_size), |_| {
                uniform.sample(&mut rng) * weight_scale
            });

        let biases = Array1::zeros(num_filters);

        Self {
            filters,
            biases,
            stride,
            padding,
            activation_function,
        }
    }

    pub fn forward(&self, input: &ArrayView3<f32>) -> (Array3<f32>, Array3<f32>) {
        let (batch_size, input_height, input_width) = input.dim();
        let (num_filters, _, kernel_height, kernel_width) = self.filters.dim();

        let output_height = ((input_height + 2 * self.padding - kernel_height) / self.stride) + 1;
        let output_width = ((input_width + 2 * self.padding - kernel_width) / self.stride) + 1;

        let mut z = Array3::zeros((num_filters, output_height, output_width));
        for f in 0..num_filters {
            for h in 0..output_height {
                for w in 0..output_width {
                    let h_start = h * self.stride;
                    let h_end = h_start + kernel_height;
                    let w_start = w * self.stride;
                    let w_end = w_start + kernel_width;

                    let mut sum = 0.0;
                    for c in 0..input.dim().0 {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = h_start + kh;
                                let iw = w_start + kw;

                                // Handle padding
                                if ih >= self.padding
                                    && ih < input_height + self.padding
                                    && iw >= self.padding
                                    && iw < input_width + self.padding
                                {
                                    sum += input[[c, ih - self.padding, iw - self.padding]]
                                        * self.filters[[f, c, kh, kw]];
                                }
                            }
                        }
                    }
                    z[[f, h, w]] = sum + self.biases[f];
                }
            }
        }

        let mut activation = z.clone();
        activation.mapv_inplace(|x| self.activation_function.activate(x));

        (z, activation)
    }

    pub fn backward(
        &self,
        delta: &Array3<f32>,
        prev_activation: &Array3<f32>,
        z: &Array3<f32>,
    ) -> (Array4<f32>, Array1<f32>, Array3<f32>) {
        let (num_filters, output_height, output_width) = delta.dim();
        let (_, input_channels, kernel_height, kernel_width) = self.filters.dim();

        let mut filter_gradients = Array4::zeros(self.filters.dim());
        let mut bias_gradients = Array1::zeros(num_filters);
        let mut prev_delta = Array3::zeros(prev_activation.dim());

        // Calculate gradients
        let z_derivative = z.mapv(|x| self.activation_function.derivative(x));
        let delta = delta * &z_derivative;

        // Calculate filter gradients
        for f in 0..num_filters {
            for c in 0..input_channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let h_start = h * self.stride;
                        let h_end = h_start + kernel_height;
                        let w_start = w * self.stride;
                        let w_end = w_start + kernel_width;

                        let input_patch =
                            prev_activation.slice(s![c, h_start..h_end, w_start..w_end]);

                        filter_gradients
                            .slice_mut(s![f, c, .., ..])
                            .add_assign(&(&input_patch * delta[[f, h, w]]));
                    }
                }
            }
            bias_gradients[f] = delta.index_axis(Axis(0), f).sum();
        }

        // Calculate previous layer's delta
        for c in 0..input_channels {
            for h in 0..prev_delta.dim().1 {
                for w in 0..prev_delta.dim().2 {
                    let mut sum = 0.0;
                    for f in 0..num_filters {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let oh = (h + self.padding - kh) / self.stride;
                                let ow = (w + self.padding - kw) / self.stride;

                                if oh * self.stride + kh == h + self.padding
                                    && ow * self.stride + kw == w + self.padding
                                    && oh < output_height
                                    && ow < output_width
                                {
                                    sum += delta[[f, oh, ow]] * self.filters[[f, c, kh, kw]];
                                }
                            }
                        }
                    }
                    prev_delta[[c, h, w]] = sum;
                }
            }
        }

        (filter_gradients, bias_gradients, prev_delta)
    }

    pub fn update_parameters(
        &mut self,
        filter_update: &Array4<f32>,
        bias_update: &Array1<f32>,
        learning_rate: f32,
    ) {
        self.filters -= &(learning_rate * filter_update);
        self.biases -= &(learning_rate * bias_update);
    }
}
