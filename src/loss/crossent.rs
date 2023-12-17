use nalgebra::SVector;

use super::LossFunction;

pub struct CrossEntropy;

impl<const N: usize> LossFunction<N> for CrossEntropy {
    fn func(
        y_out: SVector<f64, N>,
        y_test: SVector<f64, N>,
    ) -> f64 {
        -(0..N)
            .map(|i| y_test[i] * y_out[i].log2())
            .sum::<f64>()
    }

    fn grad(
        y_out: SVector<f64, N>,
        y_test: SVector<f64, N>,
    ) -> SVector<f64, N> {
        let out = (0..N).map(|i| -y_test[i] / y_out[i]);
        SVector::from_iterator(out)
    }
}
