![382830171-f6e61aed-b964-4a38-b3b3-07b0dba68b1f](https://github.com/user-attachments/assets/fd127726-39a5-4902-b348-afad105a43f6)

Neurotic is a **_work in progress_** machine-learning library for Rust.

<div align="center">
    <a href="https://crates.io/crates/neurotic">crates.io</a>
    <a href="https://docs.rs/neurotic/0.2.0/neurotic/">docs.rs</a>
</div>

## Installation

Add `neurotic` to your Cargo.toml dependencies:

```
cargo add neurotic
```

## Examples

### Defining the network architecture

Start by defining the amount of neurons in each layer, and the layers' activation functions.
```rust
use neurotic::{
    activation::ActivationFunction,
    core::NeuralNetwork,
    training::{CostFunction, NetworkTrainer},
};

let layer_sizes = &[2, 32, 16, 1]; // 2 neurons for the input layer, 32 and 16 for the hidden
layers, and 1 output neuron.
let activation_functions = &[
    ActivationFunction::Identity,
    ActivationFunction::ReLU,
    ActivationFunction::ReLU,
    ActivationFunction::Identity,
];
let cost_function = CostFunction::MeanSquaredError;

// Create a new instance of NeuralNetwork with the defined structure
let network = NeuralNetwork::new(layer_sizes, activation_functions, cost_function);
```

### Preparing the training data

Load in or generate your training data. Here is a simple example that generates training data for a sum function.

```rust
use rand::Rng;

// This returns a vector of tuples. Each tuple is made up of inputs, and target outputs.
fn generate_sum_data(size: usize, range: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::with_capacity(usize);
    for _ in 0..size {
        let a = rand::thread_rng.gen_range(0.0..range);
        let b = rand::thread_rng.gen_range(0.0..range);
        let output = a + b;
        data.push((vec![a, b], vec![output]));
    }
    data
}

// Store the generated training data in a variable
let training_data = generate_sum_data(1000, 10.0);
```

### Training the network

Set up the training parameters, and train the network using a `NetworkTrainer`.
```rust
let learning_rate = 0.001; // Network's learning rate
let batch_size = 50; // Divide the training data into batches of this size
let epochs = 500; // Number of training iterations

let mut trainer = NetworkTrainer::new(network, learning_rate, batch_size);
trainer.train(training_data, epochs);
```

### Saving or loading a network

Saving the trained network to a file.
```rust
trainer.get_network().save("path/to/file").expect("Failed to save network");
```

Loading a trained network from a file.
```rust
let network = NeuralNetwork::load("path/to/file").expect("Failed to load network");
```

## Contributing
Pull requests are the best way to propose changes to the program.

1. Fork the repo and create your branch from `main`.
2. Make your changes.
3. If your change directly affects the program's functionality, update the documentation.
4. Issue a pull request

### Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project.

### Report issues using Github's Issues tab.
I use GitHub issues to track public bugs. Report a bug by opening a new issue.

**Issue Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
