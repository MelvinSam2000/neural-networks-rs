use std::marker::PhantomData;

use nalgebra::SVector;

use crate::activation::ActivationFunction;

pub struct ActivationLayer<const N: usize, F> {
    z: SVector<f64, N>,
    f: PhantomData<F>,
}

impl<const N: usize, F> ActivationLayer<N, F>
where
    F: ActivationFunction<N>,
{
    pub fn new() -> Self {
        let z = SVector::zeros();
        let f = PhantomData;

        Self { z, f }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SVector<f64, N>,
    ) -> SVector<f64, N> {
        self.z = x;
        F::func(&self.z)
    }

    // backprop
    pub fn bp(
        &mut self,
        g: SVector<f64, N>,
    ) -> SVector<f64, N> {
        F::grad(&self.z) * g
    }
}
