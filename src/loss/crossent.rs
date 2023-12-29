use nalgebra::SVector;

use super::LossFunction;

pub struct CrossEntropy;

impl<const N: usize> LossFunction<N> for CrossEntropy {
    fn func(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> f32 {
        -(0..N)
            .map(|i| y_test[i] * y_out[i].log2())
            .sum::<f32>()
    }

    fn grad(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> SVector<f32, N> {
        let out = (0..N).map(|i| -y_test[i] / y_out[i]);
        SVector::from_iterator(out)
    }
}
