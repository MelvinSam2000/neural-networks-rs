use nalgebra::SMatrix;
use nalgebra::SVector;

use super::ActivationFunction;

pub struct Tanh;

impl ActivationFunction for Tanh {
    fn func(x: f32) -> f32 {
        x.tanh()
    }

    fn deriv(x: f32) -> f32 {
        1. - x.tanh().powi(2)
    }
}
