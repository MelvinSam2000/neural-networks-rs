use nalgebra::SMatrix;

use super::component_invsqrt;
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
            * component_invsqrt(&self.g)
                .component_mul(&gradient);
    }

    fn name() -> String {
        "adagrad".to_string()
    }
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
