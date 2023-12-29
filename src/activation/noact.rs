use super::ActivationFunction;

pub struct NoActivation;

impl ActivationFunction for NoActivation {
    fn func(x: f32) -> f32 {
        x
    }

    fn deriv(_: f32) -> f32 {
        1.
    }
}
