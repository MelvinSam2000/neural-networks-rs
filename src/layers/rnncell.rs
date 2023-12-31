use std::marker::PhantomData;

use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::ActivationFunction;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct RnnCell<
    const X: usize,
    const Y: usize,
    const H: usize,
    const T: usize,
    F,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>,
> {
    x: [SVector<f32, X>; T],
    y: [SVector<f32, Y>; T],
    h: [SVector<f32, H>; T],
    z: [SVector<f32, H>; T],
    wx: SMatrix<f32, H, X>,
    wh: SMatrix<f32, H, H>,
    wy: SMatrix<f32, Y, H>,
    optwx: <O as OptimizerFactory<H, X>>::Optimizer,
    optwh: <O as OptimizerFactory<H, H>>::Optimizer,
    optwy: <O as OptimizerFactory<Y, H>>::Optimizer,
    act: PhantomData<F>,
}

impl<
        const X: usize,
        const Y: usize,
        const H: usize,
        const T: usize,
        F,
        O,
    > RnnCell<X, Y, H, T, F, O>
where
    F: ActivationFunction,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>,
{
    pub fn new() -> Self {
        let x = [SVector::zeros(); T];
        let y = [SVector::zeros(); T];
        let h = [SVector::zeros(); T];
        let z = [SVector::zeros(); T];

        let mut wx = SMatrix::zeros();
        let mut wy = SMatrix::zeros();
        let mut wh = SMatrix::zeros();

        // randomize weights
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-1.0, 1.0);
        for i in 0..H {
            for j in 0..X {
                wx[(i, j)] = rng.sample(uniform);
            }
        }
        for i in 0..Y {
            for j in 0..H {
                wy[(i, j)] = rng.sample(uniform);
            }
        }
        for i in 0..H {
            for j in 0..H {
                wh[(i, j)] = rng.sample(uniform);
            }
        }

        let act = PhantomData;

        let optwx =
            <O as OptimizerFactory<H, X>>::Optimizer::init(
            );
        let optwh =
            <O as OptimizerFactory<H, H>>::Optimizer::init(
            );
        let optwy =
            <O as OptimizerFactory<Y, H>>::Optimizer::init(
            );

        Self {
            x,
            y,
            h,
            wx,
            wh,
            wy,
            z,
            optwx,
            optwh,
            optwy,
            act,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: [SVector<f32, X>; T],
    ) -> [SVector<f32, Y>; T] {
        self.x = x;
        for t in 0..T {
            self.z[t] = self.wx * self.x[t];
            if t != 0 {
                self.z[t] += self.wh * self.h[t - 1]
            }
            self.h[t] = func_all::<H, 1, F>(&self.z[t]);
            self.y[t] = self.wy * self.h[t];
        }
        self.y.clone()
    }

    // backprop
    pub fn bp(
        &mut self,
        gl: [SVector<f32, Y>; T],
    ) -> [SVector<f32, X>; T] {
        let wx_old = self.wx.clone();
        let wy_old = self.wy.clone();
        let wh_old = self.wh.clone();

        // update Wy
        let mut dwy = SMatrix::zeros();
        for t in 0..T {
            dwy += gl[t] * self.h[t].transpose();
        }

        // update Wx and Wh
        let mut g = SVector::zeros();
        let mut gout = [SVector::zeros(); T];
        let mut dwx = SMatrix::zeros();
        let mut dwh = SMatrix::zeros();
        for t in (0..T).rev() {
            let g_tmp = deriv_all::<H, 1, F>(&self.z[t])
                .component_mul(
                    &(wy_old.transpose() * gl[t] + &g),
                );
            dwx += g_tmp * self.x[t].transpose();
            if t != 0 {
                dwh += g_tmp * self.h[t - 1].transpose();
            }
            gout[t] = wx_old.transpose() * g_tmp;
            g += wh_old.transpose() * g_tmp;
        }

        self.optwx.update_param(&mut self.wx, &dwx);
        self.optwh.update_param(&mut self.wh, &dwh);
        self.optwy.update_param(&mut self.wy, &dwy);

        gout
    }
}
