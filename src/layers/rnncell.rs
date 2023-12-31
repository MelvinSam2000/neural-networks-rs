use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::tanh::Tanh;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct RnnCell<
    const X: usize,
    const Y: usize,
    const H: usize,
    const T: usize,
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
}

impl<
        const X: usize,
        const Y: usize,
        const H: usize,
        const T: usize,
        O,
    > RnnCell<X, Y, H, T, O>
where
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
            self.h[t] = func_all::<H, 1, Tanh>(&self.z[t]);
            self.y[t] = self.wy * self.h[t];
        }
        self.y.clone()
    }

    // backprop
    pub fn bp(
        &mut self,
        gy: [SVector<f32, Y>; T],
    ) -> [SVector<f32, X>; T] {
        let mut gh = SVector::zeros();
        let mut gx = [SVector::zeros(); T];
        let mut dwx = SMatrix::zeros();
        let mut dwy = SMatrix::zeros();
        let mut dwh = SMatrix::zeros();
        for t in (0..T).rev() {
            dwy += gy[t] * self.h[t].transpose();
            let g = self.wy.transpose() * gy[t] + &gh;
            let g = deriv_all::<H, 1, Tanh>(&self.z[t])
                .component_mul(&g);
            dwx += g * self.x[t].transpose();
            if t != 0 {
                dwh += g * self.h[t - 1].transpose();
            }
            gx[t] = self.wx.transpose() * g;
            gh = self.wh.transpose() * g;
        }

        self.optwx.update_param(&mut self.wx, &dwx);
        self.optwh.update_param(&mut self.wh, &dwh);
        self.optwy.update_param(&mut self.wy, &dwy);

        gx
    }
}
