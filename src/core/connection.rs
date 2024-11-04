use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub weight: f32,
    pub delta_weight: f32,
}

impl Connection {
    pub fn new(num_inputs: usize) -> Self {
        Self {
            weight: (rand::random::<f32>() * 2.0 - 1.0) / (num_inputs as f32).sqrt(),
            delta_weight: 0.0,
        }
    }
}
