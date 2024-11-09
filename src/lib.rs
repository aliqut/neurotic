#![warn(missing_docs)]
//! ![382830171-f6e61aed-b964-4a38-b3b3-07b0dba68b1f](https://github.com/user-attachments/assets/fd127726-39a5-4902-b348-afad105a43f6)
//!
//! `neurotic` is a library for machine-learning in Rust.
//!
//! ## Quickstart
//!
//! ### Defining the network architecture
//!
//! Start by defining the amount of neurons in each layer, and the layers' activation functions.
//! ```rust,no_run
//! use neurotic::{
//!     activation::ActivationFunction,
//!     core::NeuralNetwork,
//!     training::{CostFunction, NetworkTrainer},
//! };
//!
//! let layer_sizes = &[2, 32, 16, 1]; // 2 neurons for the input layer, 32 and 16 for the hidden
//! layers, and 1 output neuron.
//! let activation_functions = &[
//!     ActivationFunction::Identity,
//!     ActivationFunction::ReLU,
//!     ActivationFunction::ReLU,
//!     ActivationFunction::Identity,
//! ];
//! let cost_function = CostFunction::MeanSquaredError;
//!
//! // Create a new instance of NeuralNetwork with the defined structure
//! let network = NeuralNetwork::new(layer_sizes, activation_functions, cost_function);
//! ```
//!
//! ### Preparing the training data
//!
//! Load in or generate your training data. Here is a simple example that generates training data for a sum function.
//!
//! ```rust,no_run
//! use rand::Rng;
//!
//! // This returns a vector of tuples. Each tuple is made up of inputs, and target outputs.
//! fn generate_sum_data(size: usize, range: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
//!     let mut data = Vec::with_capacity(usize);
//!     for _ in 0..size {
//!         let a = rand::thread_rng.gen_range(0.0..range);
//!         let b = rand::thread_rng.gen_range(0.0..range);
//!         let output = a + b;
//!         data.push((vec![a, b], vec![output]));
//!     }
//!     data
//! }
//!
//! // Store the generated training data in a variable
//! let training_data = generate_sum_data(1000, 10.0);
//! ```
//!
//! ### Training the network
//!
//! Set up the training parameters, and train the network using a `NetworkTrainer`.
//! ```rust,no_run
//! let learning_rate = 0.001; // Network's learning rate
//! let batch_size = 50; // Divide the training data into batches of this size
//! let epochs = 500; // Number of training iterations
//!
//! let mut trainer: NetworkTrainer<NoOptimiser> = NetworkTrainer::new(network, learning_rate, batch_size, None);
//! trainer.train(training_data, epochs);
//! ```
//!
//! ### Saving or loading a network
//!
//! Saving the trained network to a file.
//! ```rust,no_run
//! trainer.get_network().save("path/to/file").expect("Failed to save network");
//! ```
//!
//! Loading a trained network from a file.
//! ```rust,no_run
//! let network = NeuralNetwork::load("path/to/file").expect("Failed to load network");
//! ```

/// Core components of the library.
pub mod core {
    mod layer;
    mod network;

    pub use layer::Layer;
    pub use network::NeuralNetwork;
}

/// Activation functions including ReLU, LeakyReLU, Sigmoid, etc.
pub mod activation {
    mod functions;

    pub use functions::ActivationFunction;
}

/// Training modules, such as the network trainer, cost functions, batching, etc.
pub mod training {
    mod batch;
    mod cost;
    mod trainer;

    pub use batch::Batch;
    pub use cost::CostFunction;
    pub use trainer::NetworkTrainer;
}

/// Network training optimisers, such as the RMSProp and Adam optimiser.
pub mod optimisers {
    mod adam;
    mod no_optimiser;
    mod optimiser;
    mod rmsprop;

    //pub use adam::Adam;
    pub use no_optimiser::NoOptimiser;
    pub use optimiser::Optimiser;
    pub use rmsprop::RMSProp;
}
