use nalgebra::SVector;

pub trait ActivationFunction<const N: usize> {
    fn func(&self, v: &SVector<f64, N>) -> SVector<f64, N>;
    fn deriv(&self, v: &SVector<f64, N>)
        -> SVector<f64, N>;
}

pub struct Sigmoid;
pub struct Relu;
pub struct Tanh;

impl Sigmoid {
    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }
}

impl<const N: usize> ActivationFunction<N> for Sigmoid {
    fn func(&self, v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(Self::sigmoid);
        SVector::from_iterator(out)
    }

    fn deriv(
        &self,
        v: &SVector<f64, N>,
    ) -> SVector<f64, N> {
        let out = v.iter().copied().map(|x| {
            let y = Self::sigmoid(x);
            y * (1. - y)
        });
        SVector::from_iterator(out)
    }
}

impl<const N: usize> ActivationFunction<N> for Relu {
    fn func(&self, v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(|x| x.max(0.));
        SVector::from_iterator(out)
    }

    fn deriv(
        &self,
        v: &SVector<f64, N>,
    ) -> SVector<f64, N> {
        let out = v.iter().copied().map(|x| {
            if x >= 0. {
                1.
            } else {
                0.
            }
        });
        SVector::from_iterator(out)
    }
}

impl<const N: usize> ActivationFunction<N> for Tanh {
    fn func(&self, v: &SVector<f64, N>) -> SVector<f64, N> {
        let out = v.iter().copied().map(f64::tanh);
        SVector::from_iterator(out)
    }

    fn deriv(
        &self,
        v: &SVector<f64, N>,
    ) -> SVector<f64, N> {
        let out = v
            .iter()
            .copied()
            .map(|x| 1. - x.tanh().powi(2));
        SVector::from_iterator(out)
    }
}
