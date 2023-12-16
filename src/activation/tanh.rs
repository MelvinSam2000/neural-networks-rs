use nalgebra::SVector;

use super::ActivationFunction;

pub struct Tanh;
impl<const N: usize> ActivationFunction<N> for Tanh {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(f64::tanh);
        SVector::from_iterator(out)
    }

    fn deriv(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v
            .iter()
            .copied()
            .map(|x| 1. - x.tanh().powi(2));
        SVector::from_iterator(out)
    }
}
