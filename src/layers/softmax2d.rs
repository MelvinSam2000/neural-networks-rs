use nalgebra::SMatrix;
use rand_distr::num_traits::Zero;

use super::softmax::Softmax;

pub struct Softmax2d<const R: usize, const C: usize> {
    softmaxs: [Softmax<C>; R],
}

impl<const R: usize, const C: usize> Softmax2d<R, C> {
    pub fn new() -> Self {
        todo!()
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f32, R, C>,
    ) -> SMatrix<f32, R, C> {
        let mut out: SMatrix<f32, R, C> = SMatrix::zero();
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
        g: SMatrix<f32, R, C>,
    ) -> SMatrix<f32, R, C> {
        let mut out: SMatrix<f32, R, C> = SMatrix::zero();
        for i in 0..R {
            let row = g.row(i).transpose();
            let row = self.softmaxs[i].bp(row);
            out.set_row(i, &row.transpose());
        }
        out
    }
}
