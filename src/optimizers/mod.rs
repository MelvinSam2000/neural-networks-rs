use nalgebra::SMatrix;

pub mod rmsprop;
pub mod sgd;
pub mod sgdmomentum;
pub trait Optimizer<const R: usize, const C: usize> {
    fn init() -> Self;
    fn update_param(
        &mut self,
        weight: &mut SMatrix<f64, R, C>,
        gradient: &SMatrix<f64, R, C>,
    );
}

pub trait OptimizerFactory<const R: usize, const C: usize> {
    type Optimizer: Optimizer<R, C>;
}
