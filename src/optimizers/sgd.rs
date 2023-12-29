use nalgebra::SMatrix;

use super::Optimizer;
use super::OptimizerFactory;

pub struct Sgd<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
>;

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C> for Sgd<ALPHA_NUM, ALPHA_DEN>
{
    fn init() -> Self {
        Self
    }

    fn update_param(
        &mut self,
        weight: &mut SMatrix<f32, R, C>,
        gradient: &SMatrix<f32, R, C>,
    ) {
        let alpha = ALPHA_NUM as f32 / ALPHA_DEN as f32;
        *weight -= alpha * gradient;
    }

    fn name() -> String {
        "SGD".to_string()
    }
}

pub struct SgdFactory<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
>;

impl<
        const R: usize,
        const C: usize,
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
    > OptimizerFactory<R, C>
    for SgdFactory<ALPHA_NUM, ALPHA_DEN>
{
    type Optimizer = Sgd<ALPHA_NUM, ALPHA_DEN>;
}
