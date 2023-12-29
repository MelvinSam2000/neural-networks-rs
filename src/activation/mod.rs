use nalgebra::SMatrix;

pub mod noact;
pub mod relu;
pub mod sigmoid;
//pub mod softmax;
pub mod tanh;

pub trait ActivationFunction {
    fn func(x: f64) -> f64;
    fn deriv(x: f64) -> f64;
}

pub fn func_all<
    const R: usize,
    const C: usize,
    F: ActivationFunction,
>(
    x: &SMatrix<f64, R, C>,
) -> SMatrix<f64, R, C> {
    let mut x = x.clone();
    x.iter_mut().for_each(|xi| *xi = F::func(*xi));
    x
}

pub fn deriv_all<
    const R: usize,
    const C: usize,
    F: ActivationFunction,
>(
    x: &SMatrix<f64, R, C>,
) -> SMatrix<f64, R, C> {
    let mut x = x.clone();
    x.iter_mut().for_each(|xi| *xi = F::deriv(*xi));
    x
}
