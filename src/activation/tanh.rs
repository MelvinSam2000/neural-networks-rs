use nalgebra::SMatrix;
use nalgebra::SVector;

use super::ActivationFunction;

pub struct Tanh;
impl<const N: usize> ActivationFunction<N> for Tanh {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(f64::tanh);
        SVector::from_iterator(out)
    }

    fn grad(v: &SVector<f64, N>) -> SMatrix<f64, N, N> {
        let out = v
            .iter()
            .copied()
            .map(|x| 1. - x.tanh().powi(2));
        SMatrix::from_diagonal(&SVector::from_iterator(out))
    }
}
