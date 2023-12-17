use nalgebra::SMatrix;
use nalgebra::SVector;

use super::ActivationFunction;

pub struct Relu;

impl<const N: usize> ActivationFunction<N> for Relu {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(|x| x.max(0.));
        SVector::from_iterator(out)
    }

    fn grad(v: &SVector<f64, N>) -> SMatrix<f64, N, N> {
        let out = v.iter().copied().map(|x| {
            if x >= 0. {
                1.
            } else {
                0.
            }
        });
        SMatrix::from_diagonal(&SVector::from_iterator(out))
    }
}
