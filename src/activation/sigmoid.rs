use nalgebra::SVector;

use super::ActivationFunction;

pub struct Sigmoid;

impl Sigmoid {
    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }
}

impl<const N: usize> ActivationFunction<N> for Sigmoid {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(Self::sigmoid);
        SVector::from_iterator(out)
    }

    fn deriv(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(|x| {
            let y = Self::sigmoid(x);
            y * (1. - y)
        });
        SVector::from_iterator(out)
    }
}
