![382830171-f6e61aed-b964-4a38-b3b3-07b0dba68b1f](https://github.com/user-attachments/assets/fd127726-39a5-4902-b348-afad105a43f6)

Neurotic is a **_work in progress_** machine-learning library for Rust.

<div align="center">
    <a href="https://crates.io/crates/neurotic">crates.io</a>
    <a href="https://docs.rs/neurotic/0.1.0/neurotic/">docs.rs</a>
</div>

## Installation

Add `neurotic` to your Cargo.toml dependencies:

```
cargo add neurotic
```

## Examples

### Creating and training a Neural Network

```rust
fn main() {
    // Network parameters
    let layer_sizes = &[2, 32, 16, 1];
    let activation_functions = &[
        ActivationFunction::Identity,
        ActivationFunction::ReLU,
        ActivationFunction::ReLU,
        ActivationFunction::Identity,
    ];
    let cost_function = CostFunction::MeanSquaredError;

    // Create the NeuralNetwork
    let network = NeuralNetwork::new(layer_sizes, activation_functions, cost_function);

    // Load in preprocessed training data
    let training_data = get_training_data();

    // Training parameters
    let learning_rate = 0.01;
    let batch_size = 32;
    let epochs = 300;

    // Training
    let mut trainer = NetworkTrainer::new(network, learning_rate, batch_size);
    trainer.train(training_data, epochs);
}
```

### Saving/loading networks to/from files

```rust
fn main () {
  // ...
  // Training code above

  // Saving the network into a file
  let trained_network = trainer.get_network();
  trained_network.save(path/to/file);

  // Loading a network from a file
  let trained_network = NeuralNetwork::load(path/to/file);
}
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
