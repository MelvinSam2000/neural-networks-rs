use nalgebra::SMatrix;

use super::component_invsqrt;
use super::Optimizer;
use super::OptimizerFactory;

pub struct RmsProp<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const RHO_NUM: usize,
    const RHO_DEN: usize,
    const R: usize,
    const C: usize,
> {
    g: SMatrix<f64, R, C>,
}

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const RHO_NUM: usize,
        const RHO_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C>
    for RmsProp<
        ALPHA_NUM,
        ALPHA_DEN,
        RHO_NUM,
        RHO_DEN,
        R,
        C,
    >
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
            + (1. - rho) * gradient.component_mul(gradient);
        *weight -= alpha
            * component_invsqrt(&self.g)
                .component_mul(&gradient);
    }

    fn name() -> String {
        "rmsprop".to_string()
    }
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
    type Optimizer = RmsProp<
        ALPHA_NUM,
        ALPHA_DEN,
        RHO_NUM,
        RHO_DEN,
        R,
        C,
    >;
}
