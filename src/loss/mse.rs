use nalgebra::SVector;

use super::LossFunction;

pub struct Mse;

impl<const N: usize> LossFunction<N> for Mse {
    fn func(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> f32 {
        (y_out - y_test).norm_squared()
    }

    fn grad(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> SVector<f32, N> {
        y_out - y_test
    }
}
