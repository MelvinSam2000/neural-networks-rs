use nalgebra::SVector;

pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub trait ActivationFunction<const N: usize> {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N>;
    fn deriv(v: &SVector<f64, N>) -> SVector<f64, N>;
}
