use std::marker::PhantomData;

use nalgebra::SMatrix;
use rand::Rng;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::ActivationFunction;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct Dense2D<
    const X: usize,
    const Y: usize,
    const N: usize,
    F,
    O: OptimizerFactory<Y, X> + OptimizerFactory<Y, N>,
> {
    x: SMatrix<f32, X, N>,
    w: SMatrix<f32, Y, X>,
    b: SMatrix<f32, Y, N>,
    z: SMatrix<f32, Y, N>,
    act: PhantomData<F>,
    optw: <O as OptimizerFactory<Y, X>>::Optimizer,
    optb: <O as OptimizerFactory<Y, N>>::Optimizer,
}

impl<
        const X: usize,
        const Y: usize,
        const N: usize,
        F,
        O,
    > Dense2D<X, Y, N, F, O>
where
    F: ActivationFunction,
    O: OptimizerFactory<Y, X> + OptimizerFactory<Y, N>,
{
    pub fn new() -> Self {
        let x = SMatrix::zeros();
        let mut w = SMatrix::zeros();
        let mut b = SMatrix::zeros();
        let z = SMatrix::zeros();

        // randomize W and b
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-0.5, 0.5);
        for i in 0..Y {
            for j in 0..X {
                w[(i, j)] = rng.sample(uniform);
            }
            b[i] = rng.sample(uniform);
        }

        let act = PhantomData;
        let optw =
            <O as OptimizerFactory<Y, X>>::Optimizer::init(
            );
        let optb =
            <O as OptimizerFactory<Y, N>>::Optimizer::init(
            );

        Self {
            x,
            w,
            b,
            z,
            act,
            optw,
            optb,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f32, X, N>,
    ) -> SMatrix<f32, Y, N> {
        self.x = x;
        self.z = self.w * self.x + self.b;
        func_all::<Y, N, F>(&self.z)
    }

    // backprop
    pub fn bp(
        &mut self,
        mut g: SMatrix<f32, Y, N>,
    ) -> SMatrix<f32, X, N> {
        g = deriv_all::<Y, N, F>(&self.z).component_mul(&g);
        let dw = &g * self.x.transpose();
        let db = &g;
        let dx = self.w.transpose() * g;
        self.optw.update_param(&mut self.w, &dw);
        self.optb.update_param(&mut self.b, db);
        dx
    }
}
