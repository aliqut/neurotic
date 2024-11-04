#[derive(Clone, Debug)]
pub struct Batch<T> {
    pub inputs: Vec<Vec<f32>>,
    pub targets: Vec<T>,
}

impl<T> Batch<T> {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<T>) -> Self {
        Self { inputs, targets }
    }
}
