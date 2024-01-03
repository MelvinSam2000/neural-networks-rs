use std::marker::PhantomData;

use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::ActivationFunction;
use crate::layers::dense::Dense;
use crate::loss::LossFunction;
use crate::optimizers::OptimizerFactory;

pub struct Ann<
    const X: usize,
    const Y: usize,
    const H: usize,
    const L: usize,
    F,
    LOSS,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, 1>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>
        + OptimizerFactory<Y, 1>,
> {
    dense: Dense<X, Y, H, L, F, O>,
    loss: PhantomData<LOSS>,
}

impl<
        const X: usize,
        const Y: usize,
        const H: usize,
        const L: usize,
        F,
        LOSS,
        O,
    > NeuralNetwork<Y> for Ann<X, Y, H, L, F, LOSS, O>
where
    F: ActivationFunction,
    LOSS: LossFunction<Y>,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, 1>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>
        + OptimizerFactory<Y, 1>,
{
    type ModelInput = SVector<f32, X>;

    fn new() -> Self {
        let dense = Dense::new();
        let loss = PhantomData;
        Self { dense, loss }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, Y> {
        self.dense.ff(x)
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, Y>,
        y_test: SVector<f32, Y>,
    ) {
        let g = LOSS::grad(y_out, y_test);
        self.dense.bp(g);
    }

    fn loss(
        y_out: &SVector<f32, Y>,
        y_test: &SVector<f32, Y>,
    ) -> f32 {
        LOSS::func(y_out.clone(), y_test.clone())
    }
}
