use nalgebra::SMatrix;
use rand_distr::num_traits::Zero;

use crate::activation::softmax::Softmax;
use crate::layers::actlayer::ActivationLayer;

pub struct Softmax2d<const R: usize, const C: usize> {
    softmaxs: [ActivationLayer<C, Softmax>; R],
}

impl<const R: usize, const C: usize> Softmax2d<R, C> {
    pub fn new() -> Self {
        todo!()
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f64, R, C>,
    ) -> SMatrix<f64, R, C> {
        let mut out: SMatrix<f64, R, C> = SMatrix::zero();
        for i in 0..R {
            let row = x.row(i).transpose();
            let row = self.softmaxs[i].ff(row);
            out.set_row(i, &row.transpose());
        }
        out
    }

    // backprop
    pub fn bp(
        &mut self,
        g: SMatrix<f64, R, C>,
    ) -> SMatrix<f64, R, C> {
        let mut out: SMatrix<f64, R, C> = SMatrix::zero();
        for i in 0..R {
            let row = g.row(i).transpose();
            let row = self.softmaxs[i].bp(row);
            out.set_row(i, &row.transpose());
        }
        out
    }
}
