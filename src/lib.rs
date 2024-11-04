pub mod core {
    mod connection;
    mod layer;
    mod network;
    mod neuron;

    pub use connection::Connection;
    pub use network::NeuralNetwork;
}

pub mod activation {
    mod functions;

    pub use functions::ActivationFunction;
}

pub mod training {
    mod batch;
    mod cost;
    mod trainer;

    pub use batch::Batch;
    pub use cost::CostFunction;
    pub use trainer::NetworkTrainer;
}
