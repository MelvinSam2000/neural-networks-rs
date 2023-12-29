use super::ActivationFunction;

pub struct Relu;

impl ActivationFunction for Relu {
    fn func(x: f64) -> f64 {
        x.max(0.)
    }

    fn deriv(x: f64) -> f64 {
        if x >= 0. {
            1.
        } else {
            0.
        }
    }
}
