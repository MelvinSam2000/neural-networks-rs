use nalgebra::SMatrix;
use rand::Rng;

use super::softmax2d::Softmax2d;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct Attention<
    const M: usize,
    const N: usize,
    const D: usize,
    O: OptimizerFactory<M, D> + OptimizerFactory<M, M>,
> {
    x: SMatrix<f64, N, M>,
    wk: SMatrix<f64, M, D>,
    wq: SMatrix<f64, M, D>,
    wv: SMatrix<f64, M, M>,
    k: SMatrix<f64, N, D>,
    q: SMatrix<f64, N, D>,
    v: SMatrix<f64, N, M>,
    z: SMatrix<f64, N, N>,
    s: SMatrix<f64, N, N>,
    softmax2d: Softmax2d<N, N>,
    optkq: <O as OptimizerFactory<M, D>>::Optimizer,
    optv: <O as OptimizerFactory<M, M>>::Optimizer,
}

impl<const M: usize, const N: usize, const D: usize, O>
    Attention<M, N, D, O>
where
    O: OptimizerFactory<M, D> + OptimizerFactory<M, M>,
{
    pub fn new() -> Self {
        let x = SMatrix::zeros();
        let mut wk = SMatrix::zeros();
        let mut wq = SMatrix::zeros();
        let mut wv = SMatrix::zeros();
        let k = SMatrix::zeros();
        let q = SMatrix::zeros();
        let v = SMatrix::zeros();
        let z = SMatrix::zeros();
        let s = SMatrix::zeros();

        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-0.5, 0.5);
        for i in 0..M {
            for j in 0..D {
                wk[(i, j)] = rng.sample(uniform);
                wq[(i, j)] = rng.sample(uniform);
            }
            for j in 0..M {
                wv[(i, j)] = rng.sample(uniform);
            }
        }

        let softmax2d = Softmax2d::new();

        let optkq =
            <O as OptimizerFactory<M, D>>::Optimizer::init(
            );
        let optv =
            <O as OptimizerFactory<M, M>>::Optimizer::init(
            );

        Self {
            x,
            wk,
            wq,
            wv,
            k,
            q,
            v,
            z,
            s,
            softmax2d,
            optkq,
            optv,
        }
    }

    pub fn ff(
        &mut self,
        x: SMatrix<f64, N, M>,
    ) -> SMatrix<f64, N, M> {
        self.x = x;
        self.k = self.x * self.wk;
        self.q = self.x * self.wq;
        self.v = self.x * self.wv;
        self.z = self.q * self.k.transpose();
        self.z = self.z / (D as f64).sqrt();
        self.s = self.softmax2d.ff(self.s);
        self.s * self.v
    }

    pub fn bp(
        &mut self,
        g: SMatrix<f64, N, M>,
    ) -> SMatrix<f64, N, M> {
        let gs = &g * self.v.transpose();
        // v path
        let gv = self.s.transpose() * &g;
        let dj_dwv = self.x.transpose() * &gv;
        let dj_dv = &gv * self.wv.transpose();
        // k and q path
        let gs = self.softmax2d.bp(gs);
        let gs = gs / (D as f64).sqrt();
        let gk = gs.transpose() * self.q;
        let gq = gs * self.k;
        let dj_dwk = self.x.transpose() * &gk;
        let dj_dwq = self.x.transpose() * &gq;
        let dj_dk = &gk * self.wk.transpose();
        let dj_dq = &gq * self.wq.transpose();

        self.optkq.update_param(&mut self.wk, &dj_dwk);
        self.optkq.update_param(&mut self.wq, &dj_dwq);
        self.optv.update_param(&mut self.wv, &dj_dwv);

        dj_dk + dj_dq + dj_dv
    }
}
