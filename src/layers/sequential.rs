use std::marker::PhantomData;

use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::ActivationFunction;

pub struct Sequential<const L1: usize, const L2: usize, F> {
    a: SVector<f64, L1>,
    w: SMatrix<f64, L2, L1>,
    b: SVector<f64, L2>,
    z: SVector<f64, L2>,
    learn_rate: f64,
    act: PhantomData<F>,
}

impl<const L1: usize, const L2: usize, F>
    Sequential<L1, L2, F>
where
    F: ActivationFunction<L2>,
{
    pub fn new(learn_rate: f64) -> Self {
        let a = SVector::zeros();
        let mut w = SMatrix::zeros();
        let mut b = SVector::zeros();
        let z = SVector::zeros();

        // randomize W and b
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-1.0, 1.0);
        for i in 0..L2 {
            for j in 0..L1 {
                w[(i, j)] = rng.sample(uniform);
            }
            b[i] = rng.sample(uniform);
        }

        let act = PhantomData;

        Self {
            a,
            w,
            b,
            z,
            learn_rate,
            act,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        a: SVector<f64, L1>,
    ) -> SVector<f64, L2> {
        self.a = a;
        self.z = self.w * self.a + self.b;
        F::func(&self.z)
    }

    // backprop
    pub fn bp(
        &mut self,
        mut g: SVector<f64, L2>,
    ) -> SVector<f64, L1> {
        let dadz =
            SMatrix::from_diagonal(&F::deriv(&self.z));
        g = dadz * g;
        let w_copy = self.w.clone();
        let dzdw = &g * self.a.transpose();
        self.w -= self.learn_rate * dzdw;
        self.b -= self.learn_rate * &g;
        let dzda = w_copy.transpose();
        dzda * g
    }
}
