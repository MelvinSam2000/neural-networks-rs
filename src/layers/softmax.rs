use nalgebra::SMatrix;
use nalgebra::SVector;

pub struct Softmax<const N: usize> {
    s: SVector<f64, N>,
}

impl<const N: usize> Softmax<N> {
    pub fn new() -> Self {
        let s = SVector::zeros();
        Self { s }
    }

    pub fn ff(
        &mut self,
        x: SVector<f64, N>,
    ) -> SVector<f64, N> {
        let sum_exp: f64 =
            x.iter().copied().map(|xi| xi.exp()).sum();
        let out =
            x.iter().copied().map(|xi| xi.exp() / sum_exp);
        self.s = SVector::from_iterator(out);
        self.s
    }

    pub fn bp(
        &mut self,
        g: SVector<f64, N>,
    ) -> SVector<f64, N> {
        let mut ds: SMatrix<f64, N, N> = SMatrix::zeros();
        for i in 0..N {
            for j in 0..N {
                ds[(i, j)] = if i == j {
                    self.s[i] * (1. - self.s[i])
                } else {
                    -self.s[i] * self.s[j]
                };
            }
        }
        ds * g
    }
}
