use std::marker::PhantomData;

use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::ActivationFunction;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct Sequential<
    const L1: usize,
    const L2: usize,
    F,
    O: OptimizerFactory<L2, L1> + OptimizerFactory<L2, 1>,
> {
    a: SVector<f64, L1>,
    w: SMatrix<f64, L2, L1>,
    b: SVector<f64, L2>,
    z: SVector<f64, L2>,
    act: PhantomData<F>,
    optw: <O as OptimizerFactory<L2, L1>>::Optimizer,
    optb: <O as OptimizerFactory<L2, 1>>::Optimizer,
}

impl<const L1: usize, const L2: usize, F, O>
    Sequential<L1, L2, F, O>
where
    F: ActivationFunction,
    O: OptimizerFactory<L2, L1> + OptimizerFactory<L2, 1>,
{
    pub fn new() -> Self {
        let a = SVector::zeros();
        let mut w = SMatrix::zeros();
        let mut b = SVector::zeros();
        let z = SVector::zeros();

        // randomize W and b
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-0.5, 0.5);
        for i in 0..L2 {
            for j in 0..L1 {
                w[(i, j)] = rng.sample(uniform);
            }
            b[i] = rng.sample(uniform);
        }

        let act = PhantomData;
        let optw = <O as OptimizerFactory<L2, L1>>::Optimizer::init();
        let optb =
            <O as OptimizerFactory<L2, 1>>::Optimizer::init(
            );

        Self {
            a,
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
        a: SVector<f64, L1>,
    ) -> SVector<f64, L2> {
        self.a = a;
        self.z = self.w * self.a + self.b;
        func_all::<L2, 1, F>(&self.z)
    }

    // backprop
    pub fn bp(
        &mut self,
        mut g: SVector<f64, L2>,
    ) -> SVector<f64, L1> {
        g = deriv_all::<L2, 1, F>(&self.z)
            .component_mul(&g);
        let dzdw = &g * self.a.transpose();
        let dzda = self.w.transpose();
        self.optw.update_param(&mut self.w, &dzdw);
        self.optb.update_param(&mut self.b, &g);
        dzda * g
    }
}
