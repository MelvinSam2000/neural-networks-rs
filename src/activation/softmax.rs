use nalgebra::SMatrix;
use nalgebra::SVector;

use super::ActivationFunction;

pub struct Softmax;

impl<const N: usize> ActivationFunction<N> for Softmax {
    fn func(v: &SVector<f64, N>) -> SVector<f64, N> {
        let sum_exp: f64 =
            v.iter().copied().map(|x| x.exp()).sum();
        let out =
            v.iter().copied().map(|x| x.exp() / sum_exp);
        SVector::from_iterator(out)
    }

    fn grad(v: &SVector<f64, N>) -> SMatrix<f64, N, N> {
        let mut out: SMatrix<f64, N, N> = SMatrix::zeros();
        let s = Self::func(v);
        for i in 0..N {
            for j in 0..N {
                out[(i, j)] = if i == j {
                    s[i] * (1. - s[i])
                } else {
                    -s[i] * s[j]
                };
            }
        }
        out
    }
}
