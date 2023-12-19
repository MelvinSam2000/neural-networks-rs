use std::marker::PhantomData;

use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::ActivationFunction;
use crate::layers::sequential::Sequential;
use crate::loss::LossFunction;

pub struct Ann4<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
    F1,
    F2,
    F3,
    LOSS,
> {
    s1: Sequential<L1, L2, F1>,
    s2: Sequential<L2, L3, F2>,
    s3: Sequential<L3, L4, F3>,
    loss: PhantomData<LOSS>,
}

impl<
        const L1: usize,
        const L2: usize,
        const L3: usize,
        const L4: usize,
        F1,
        F2,
        F3,
        LOSS,
    > NeuralNetwork<L4>
    for Ann4<L1, L2, L3, L4, F1, F2, F3, LOSS>
where
    F1: ActivationFunction<L2>,
    F2: ActivationFunction<L3>,
    F3: ActivationFunction<L4>,
    LOSS: LossFunction<L4>,
{
    type ModelInput = SVector<f64, L1>;

    fn new(learn_rate: f64) -> Self {
        let s1 = Sequential::new(learn_rate);
        let s2 = Sequential::new(learn_rate);
        let s3 = Sequential::new(learn_rate);
        let loss = PhantomData;
        Self { s1, s2, s3, loss }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f64, L4> {
        let a = x;
        let a = self.s1.ff(a);
        let a = self.s2.ff(a);
        let a = self.s3.ff(a);
        a
    }

    fn backprop(
        &mut self,
        y_out: SVector<f64, L4>,
        y_test: SVector<f64, L4>,
    ) {
        let g = LOSS::grad(y_out, y_test);
        let g = self.s3.bp(g);
        let g = self.s2.bp(g);
        self.s1.bp(g);
    }

    fn loss(
        y_out: &SVector<f64, L4>,
        y_test: &SVector<f64, L4>,
    ) -> f64 {
        LOSS::func(y_out.clone(), y_test.clone())
    }
}

impl<
        const L1: usize,
        const L2: usize,
        const L3: usize,
        const L4: usize,
        F1,
        F2,
        F3,
        LOSS,
    > Ann4<L1, L2, L3, L4, F1, F2, F3, LOSS>
where
    F1: ActivationFunction<L2>,
    F2: ActivationFunction<L3>,
    F3: ActivationFunction<L4>,
    LOSS: LossFunction<L4>,
{
    pub fn preprocess(
        x: &[[f64; L1]],
        y: &[usize],
    ) -> (Vec<SVector<f64, L1>>, Vec<SVector<f64, L4>>)
    {
        let x: Vec<SVector<f64, L1>> = x
            .iter()
            .map(|x| SVector::from_column_slice(x))
            .collect();
        let y: Vec<SVector<f64, L4>> = y
            .iter()
            .map(|&y| {
                let mut y_new = SVector::<f64, L4>::zeros();
                y_new[y] = 1.;
                y_new
            })
            .collect();
        (x, y)
    }
}
