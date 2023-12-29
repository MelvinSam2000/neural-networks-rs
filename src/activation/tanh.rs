use nalgebra::SMatrix;
use nalgebra::SVector;

use super::ActivationFunction;

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn func(x: f64) -> f64 {
        x.tanh()
    }

    fn deriv(x: f64) -> f64 {
        1. - x.tanh().powi(2)
    }
}
