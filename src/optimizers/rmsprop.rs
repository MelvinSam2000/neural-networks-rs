use nalgebra::SMatrix;
use rand_distr::num_traits::Inv;

use super::Optimizer;
use super::OptimizerFactory;

pub struct RmsProp<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const RHO_NUM: usize,
    const RHO_DEN: usize,
    const R: usize,
> {
    g: SMatrix<f64, R, R>,
}

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const RHO_NUM: usize,
        const RHO_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C>
    for RmsProp<ALPHA_NUM, ALPHA_DEN, RHO_NUM, RHO_DEN, R>
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
        let rho = RHO_NUM as f64 / RHO_DEN as f64;
        self.g = rho * &self.g
            + (1. - rho) * gradient * gradient.transpose();
        let diag = self.g.diagonal() * SMatrix::identity();
        *weight -=
            alpha * elementwise_invsqrt(&diag) * gradient;
    }
}

fn elementwise_invsqrt<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
) -> SMatrix<f64, R, C> {
    const EPSILON: f64 = 0.00000001;

    let mut out = m.clone();
    out.iter_mut().for_each(|x| {
        *x = (x.abs() + EPSILON).sqrt().inv()
    });
    out
}

pub struct RmsPropFactory<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const RHO_NUM: usize,
    const RHO_DEN: usize,
>;

impl<
        const R: usize,
        const C: usize,
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const RHO_NUM: usize,
        const RHO_DEN: usize,
    > OptimizerFactory<R, C>
    for RmsPropFactory<
        ALPHA_NUM,
        ALPHA_DEN,
        RHO_NUM,
        RHO_DEN,
    >
{
    type Optimizer =
        RmsProp<ALPHA_NUM, ALPHA_DEN, RHO_NUM, RHO_DEN, R>;
}

/*
impl Optimizer for OptRmsProp {
    fn update_param<const R: usize, const C: usize>(
        &mut self,
        weight: &mut SMatrix<f64, R, C>,
        gradient: &SMatrix<f64, R, C>,
    ) {
        self.g = self.rho * &self.g;
        //+ (1. - self.rho)
        //    * gradient
        //    * gradient.transpose();
        *weight -= self.alpha * &self.g * gradient;
    }
}
*/
