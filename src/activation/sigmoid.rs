use super::ActivationFunction;

pub struct Sigmoid;

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

impl ActivationFunction for Sigmoid {
    fn func(x: f32) -> f32 {
        sigmoid(x)
    }

    fn deriv(x: f32) -> f32 {
        let x = sigmoid(x);
        x * (1. - x)
    }
}
