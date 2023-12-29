use nalgebra::SMatrix;

pub mod adagrad;
pub mod adam;
pub mod rmsprop;
pub mod sgd;
pub mod sgdmomentum;

pub trait Optimizer<const R: usize, const C: usize> {
    fn init() -> Self;
    fn update_param(
        &mut self,
        weight: &mut SMatrix<f32, R, C>,
        gradient: &SMatrix<f32, R, C>,
    );
    fn name() -> String;
}

pub trait OptimizerFactory<const R: usize, const C: usize> {
    type Optimizer: Optimizer<R, C>;
}

pub fn component_invsqrt<const R: usize, const C: usize>(
    m: &SMatrix<f32, R, C>,
) -> SMatrix<f32, R, C> {
    const EPSILON: f32 = 0.000001;

    let mut out = m.clone();
    out.iter_mut().for_each(|x| {
        *x = 1. / (*x + EPSILON).sqrt();
    });
    out
}
