use nalgebra::SMatrix;
use rand_distr::num_traits::Inv;

use super::Optimizer;
use super::OptimizerFactory;

pub struct Adagrad<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const R: usize,
    const C: usize,
> {
    g: SMatrix<f64, R, C>,
}

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C>
    for Adagrad<ALPHA_NUM, ALPHA_DEN, R, C>
{
    fn init() -> Self {
        let g = SMatrix::zeros();
        Self { g }
    }

    fn update_param(
        &mut self,
        weight: &mut SMatrix<f64, R, C>,
        gradient: &SMatrix<f64, R, C>,
    ) {
        let alpha = ALPHA_NUM as f64 / ALPHA_DEN as f64;
        self.g += gradient.component_mul(gradient);
        *weight -= alpha
            * elementwise_invsqrt(&self.g)
                .component_mul(&gradient);
    }
}

fn elementwise_invsqrt<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
) -> SMatrix<f64, R, C> {
    const EPSILON: f64 = 0.0000001;

    let mut out = m.clone();
    out.iter_mut()
        .for_each(|x| *x = (*x + EPSILON).sqrt().inv());
    out
}

pub struct AdagradFactory<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
>;

impl<
        const R: usize,
        const C: usize,
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
    > OptimizerFactory<R, C>
    for AdagradFactory<ALPHA_NUM, ALPHA_DEN>
{
    type Optimizer = Adagrad<ALPHA_NUM, ALPHA_DEN, R, C>;
}
