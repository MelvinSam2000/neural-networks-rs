use super::ActivationFunction;

pub struct Relu;

impl ActivationFunction for Relu {
    fn func(x: f32) -> f32 {
        x.max(0.)
    }

    fn deriv(x: f32) -> f32 {
        if x >= 0. {
            1.
        } else {
            0.
        }
    }
}
