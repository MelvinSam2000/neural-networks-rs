use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::ActivationFunction;

pub struct RnnCell<
    const X: usize,
    const Y: usize,
    const HI: usize,
    const HF: usize,
    F,
> {
    x: SVector<f64, X>,
    wx: SMatrix<f64, HF, X>,
    hi: SVector<f64, HI>,
    wh: SMatrix<f64, HF, HI>,
    z: SVector<f64, HF>,
    hf: SVector<f64, HF>,
    y: SVector<f64, Y>,
    wy: SMatrix<f64, Y, HF>,
    learn_rate: f64,
    act: F,
}

impl<
        const X: usize,
        const Y: usize,
        const HI: usize,
        const HF: usize,
        F,
    > RnnCell<X, Y, HI, HF, F>
where
    F: ActivationFunction<HF>,
{
    pub fn new(
        learn_rate: f64,
        activation_function: F,
    ) -> Self {
        let x = SVector::zeros();
        let y = SVector::zeros();
        let hi = SVector::zeros();
        let hf = SVector::zeros();
        let z = SVector::zeros();

        let mut wx = SMatrix::zeros();
        let mut wy = SMatrix::zeros();
        let mut wh = SMatrix::zeros();

        // randomize weights
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-1.0, 1.0);
        for i in 0..HF {
            for j in 0..X {
                wx[(i, j)] = rng.sample(uniform);
            }
        }
        for i in 0..Y {
            for j in 0..HF {
                wy[(i, j)] = rng.sample(uniform);
            }
        }
        for i in 0..HF {
            for j in 0..HI {
                wh[(i, j)] = rng.sample(uniform);
            }
        }

        let act = activation_function;

        Self {
            x,
            wx,
            hi,
            wh,
            z,
            hf,
            y,
            wy,
            learn_rate,
            act,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SVector<f64, X>,
        hi: SVector<f64, HI>,
    ) -> (SVector<f64, Y>, SVector<f64, HF>) {
        self.x = x;
        self.hi = hi;
        self.z = self.wx * self.x + self.wh * self.hi;
        self.hf = self.act.func(&self.z);
        self.y = self.wy * self.hf;
        (self.y, self.hf)
    }

    // backprop
    pub fn bp(
        &mut self,
        gy: SVector<f64, Y>,
        gh: SVector<f64, HF>,
    ) -> SVector<f64, HI> {
        let wy_copy = self.wy.clone();
        let wh_copy = self.wh.clone();
        self.wy -=
            self.learn_rate * gy * self.hf.transpose();
        let gy = wy_copy.transpose() * gy;
        let g = gy + gh;
        let dhfdz = SMatrix::from_diagonal(
            &self.act.deriv(&self.z),
        );
        let g = dhfdz * g;
        let gh = wh_copy.transpose() * &g;
        self.wx -= self.learn_rate * g * self.x.transpose();
        self.wh -=
            self.learn_rate * g * self.hi.transpose();
        gh
    }
}
