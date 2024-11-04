/// Struct representing a batch of training data.
#[derive(Clone, Debug)]
pub struct Batch<T> {
    pub inputs: Vec<Vec<f32>>,
    pub targets: Vec<T>,
}

impl<T> Batch<T> {
    /// Constructs a new Batch.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector containing input data.
    /// * `targets` - Vector containing expected/target outputs.
    ///
    /// # Returns
    ///
    /// A new instance of Batch.
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<T>) -> Self {
        Self { inputs, targets }
    }
}
