use nalgebra::SVector;

pub mod crossent;
pub mod mse;

pub trait LossFunction<const N: usize> {
    fn func(
        y_out: SVector<f64, N>,
        y_test: SVector<f64, N>,
    ) -> f64;
    fn grad(
        y_out: SVector<f64, N>,
        y_test: SVector<f64, N>,
    ) -> SVector<f64, N>;
}
