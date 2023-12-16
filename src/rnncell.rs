use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::ActivationFunction;

pub struct RnnCell<
    const X: usize,
    const Y: usize,
    const H: usize,
    const T: usize,
    F,
> {
    x: [SVector<f64, X>; T],
    y: [SVector<f64, Y>; T],
    h: [SVector<f64, H>; T],
    z: [SVector<f64, H>; T],
    wx: SMatrix<f64, H, X>,
    wh: SMatrix<f64, H, H>,
    wy: SMatrix<f64, Y, H>,
    learn_rate: f64,
    act: F,
}

impl<
        const X: usize,
        const Y: usize,
        const H: usize,
        const T: usize,
        F,
    > RnnCell<X, Y, H, T, F>
where
    F: ActivationFunction<H>,
{
    pub fn new(
        learn_rate: f64,
        activation_function: F,
    ) -> Self {
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

        let act = activation_function;

        Self {
            x,
            y,
            h,
            wx,
            wh,
            wy,
            z,
            learn_rate,
            act,
        }
    }

    // feedforward
    pub fn ff(&mut self, x: [SVector<f64, X>; T]) {
        self.x = x;
        for t in 0..T {
            self.z[t] = self.wx * self.x[t]
                + if t != 0 {
                    self.wh[t] * self.h[t - 1]
                } else {
                    SVector::zeros()
                };
            self.h[t] = self.act.func(&self.z[t]);
            self.y[t] = self.wy * self.h[t];
        }
    }

    // backprop
    pub fn bp(&mut self, gl: [SVector<f64, Y>; T]) {
        let wy_old = self.wy.clone();
        let wh_old = self.wh.clone();

        // update Wy
        for t in 0..T {
            self.wy -= self.learn_rate
                * gl[t]
                * self.h[t].transpose();
        }

        // update Wx and Wh
        let mut g = SVector::zeros();
        for t in (0..T).rev() {
            let dht_dzt = SMatrix::from_diagonal(
                &self.act.deriv(&self.z[t]),
            );
            let g_tmp =
                dht_dzt * (wy_old.transpose() * gl[t] + &g);
            self.wx -= self.learn_rate
                * g_tmp
                * self.x[t].transpose();
            if t != 0 {
                self.wh -= self.learn_rate
                    * g_tmp
                    * self.h[t - 1].transpose();
            }
            g += wh_old.transpose() * g_tmp;
        }
    }
}
