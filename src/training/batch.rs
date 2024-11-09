/// Struct representing a batch of training data.
#[derive(Clone, Debug)]
pub struct Batch {
    /// Input data for training. Inner vector contains training samples.
    pub inputs: Vec<Vec<f32>>,
    /// Input labels for training. Inner vector contains training labels.
    pub targets: Vec<Vec<f32>>,
}

impl Batch {
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
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        Self { inputs, targets }
    }
}
