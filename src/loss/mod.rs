use nalgebra::SVector;

pub mod crossent;
pub mod mse;

pub trait LossFunction<const N: usize> {
    fn func(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> f32;
    fn grad(
        y_out: SVector<f32, N>,
        y_test: SVector<f32, N>,
    ) -> SVector<f32, N>;
}
