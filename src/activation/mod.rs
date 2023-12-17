use nalgebra::SMatrix;
use nalgebra::SVector;

pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub trait ActivationFunction<const N: usize> {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N>;
    fn grad(v: &SVector<f64, N>) -> SMatrix<f64, N, N>;
}
