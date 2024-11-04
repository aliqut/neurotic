use serde::{Deserialize, Serialize};

use super::Connection;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub activation: f32,
    pub connections: Vec<Connection>,
    pub bias: f32,
    pub delta_bias: f32,
}

impl Neuron {
    pub fn new() -> Self {
        Self {
            activation: 0.0,
            connections: Vec::new(),
            bias: 0.0,
            delta_bias: 0.0,
        }
    }
}
