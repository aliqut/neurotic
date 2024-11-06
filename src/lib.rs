//! # Neurotic
//!
//! `neurotic` is a library for machine-learning in Rust.

/// Core components of the library.
pub mod core {
    mod layer;
    mod network;

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
