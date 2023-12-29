use super::ActivationFunction;

pub struct Sigmoid;

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

impl ActivationFunction for Sigmoid {
    fn func(x: f64) -> f64 {
        sigmoid(x)
    }

    fn deriv(x: f64) -> f64 {
        let x = sigmoid(x);
        x * (1. - x)
    }
}
