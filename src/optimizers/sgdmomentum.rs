use nalgebra::SMatrix;

use super::Optimizer;
use super::OptimizerFactory;

pub struct SgdWMomentum<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const BETA_NUM: usize,
    const BETA_DEN: usize,
    const R: usize,
    const C: usize,
> {
    v: SMatrix<f32, R, C>,
}

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const BETA_NUM: usize,
        const BETA_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C>
    for SgdWMomentum<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA_NUM,
        BETA_DEN,
        R,
        C,
    >
{
    fn init() -> Self {
        let v = SMatrix::zeros();
        Self { v }
    }

    fn update_param(
        &mut self,
        weight: &mut SMatrix<f32, R, C>,
        gradient: &SMatrix<f32, R, C>,
    ) {
        let alpha = ALPHA_NUM as f32 / ALPHA_DEN as f32;
        let beta = BETA_NUM as f32 / BETA_DEN as f32;
        self.v = beta * &self.v + (1. - beta) * gradient;
        *weight -= alpha * &self.v;
    }

    fn name() -> String {
        "SGD with momentum".to_string()
    }
}

pub struct SgdWMomentumFactory<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const BETA_NUM: usize,
    const BETA_DEN: usize,
>;

impl<
        const R: usize,
        const C: usize,
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const BETA_NUM: usize,
        const BETA_DEN: usize,
    > OptimizerFactory<R, C>
    for SgdWMomentumFactory<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA_NUM,
        BETA_DEN,
    >
{
    type Optimizer = SgdWMomentum<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA_NUM,
        BETA_DEN,
        R,
        C,
    >;
}
