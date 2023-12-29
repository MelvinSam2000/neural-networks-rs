use std::marker::PhantomData;

use nalgebra::SVector;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::ActivationFunction;

pub struct ActivationLayer<const N: usize, F> {
    z: SVector<f32, N>,
    f: PhantomData<F>,
}

impl<const N: usize, F> ActivationLayer<N, F>
where
    F: ActivationFunction,
{
    pub fn new() -> Self {
        let z = SVector::zeros();
        let f = PhantomData;

        Self { z, f }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SVector<f32, N>,
    ) -> SVector<f32, N> {
        self.z = x;
        func_all::<N, 1, F>(&self.z)
    }

    // backprop
    pub fn bp(
        &mut self,
        g: SVector<f32, N>,
    ) -> SVector<f32, N> {
        deriv_all::<N, 1, F>(&self.z).component_mul(&g)
    }
}
