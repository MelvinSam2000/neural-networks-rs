use nalgebra::SMatrix;
use nalgebra::SVector;
use nalgebra::Vector1;

use super::NeuralNetwork;
use crate::activation::noact::NoActivation;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::tanh::Tanh;
use crate::layers::randembedding::RandEmbedding;
use crate::layers::rnncell::RnnCell;
use crate::layers::sequential::Sequential;
use crate::layers::softmax::Softmax;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::adam::AdamFactory;
use crate::optimizers::OptimizerFactory;

const H: usize = 1;

// N is number of words, X is dim of word embedding, Y is sentiment dimensions
pub struct RnnSentimentAnalyzer<
    const N: usize,
    const X: usize,
    const Y: usize,
    O: OptimizerFactory<100, N>
        + OptimizerFactory<100, 1>
        + OptimizerFactory<Y, 100>
        + OptimizerFactory<Y, 1>,
> {
    embedding: RandEmbedding<N, X>,
    rnn: RnnCell<
        X,
        1,
        H,
        N,
        Tanh,
        AdamFactory<1, 100, 9, 10, 9, 10>,
    >,
    s1: Sequential<N, 100, Sigmoid, O>,
    s2: Sequential<100, Y, NoActivation, O>,
    softmax: Softmax<Y>,
}

impl<const N: usize, const X: usize, const Y: usize, O>
    NeuralNetwork<Y> for RnnSentimentAnalyzer<N, X, Y, O>
where
    O: OptimizerFactory<100, N>
        + OptimizerFactory<100, 1>
        + OptimizerFactory<Y, 100>
        + OptimizerFactory<Y, 1>,
{
    type ModelInput = String;

    fn new() -> Self {
        let embedding = RandEmbedding::default();
        let rnn = RnnCell::new();
        let s1 = Sequential::new();
        let s2 = Sequential::new();
        let softmax = Softmax::new();
        Self {
            embedding,
            rnn,
            s1,
            s2,
            softmax,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, Y> {
        let x = self.embedding.embed(x);
        let x = mat_to_array(x);
        let x = self.rnn.ff(x);
        let x = flatten(x);
        let x = self.s1.ff(x);
        let x = self.s2.ff(x);
        let x = self.softmax.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, Y>,
        y_test: SVector<f32, Y>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.softmax.bp(g);
        let g = self.s2.bp(g);
        let g = self.s1.bp(g);
        let g = unflatten(g);
        self.rnn.bp(g);
    }

    fn loss(
        y_out: &SVector<f32, Y>,
        y_test: &SVector<f32, Y>,
    ) -> f32 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten<const T: usize>(
    v: [SVector<f32, 1>; T],
) -> SVector<f32, T> {
    let mut out = SVector::zeros();
    for t in 0..T {
        out[t] = v[t][0];
    }
    out
}

fn unflatten<const T: usize>(
    v: SVector<f32, T>,
) -> [SVector<f32, 1>; T] {
    let mut out = [SVector::zeros(); T];
    for t in 0..T {
        out[t] = Vector1::new(v[t]);
    }
    out
}

fn mat_to_array<const N: usize, const X: usize>(
    m: SMatrix<f32, N, X>,
) -> [SVector<f32, X>; N] {
    let mut out = [SVector::zeros(); N];
    m.row_iter().enumerate().for_each(|(i, row)| {
        out[i] = row.transpose();
    });
    out
}
