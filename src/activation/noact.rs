use super::ActivationFunction;

pub struct NoActivation;

impl ActivationFunction for NoActivation {
    fn func(x: f64) -> f64 {
        x
    }

    fn deriv(_: f64) -> f64 {
        1.
    }
}
