use nalgebra::SVector;

pub struct Embedding<const N: usize> {
    // todo
}

impl<const N: usize> Embedding<N> {
    pub fn new() -> Self {
        Self {}
    }

    pub fn embed<const T: usize>(
        &mut self,
        words: String,
    ) -> [SVector<f64, T>; N] {
        todo!()
    }
}
