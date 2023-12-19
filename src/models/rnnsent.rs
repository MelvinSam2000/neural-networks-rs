use nalgebra::SVector;
use nalgebra::Vector1;

use super::NeuralNetwork;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::softmax::Softmax;
use crate::activation::tanh::Tanh;
use crate::layers::embedding::Embedding;
use crate::layers::rnncell::RnnCell;
use crate::layers::sequential::Sequential;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;

const H: usize = 1;

// N is number of words, X is dim of word embedding, Y is sentiment dimensions
pub struct RnnSentimentAnalyzer<
    const N: usize,
    const X: usize,
    const Y: usize,
> {
    embedding: Embedding<N>,
    rnn: RnnCell<X, 1, H, N, Tanh>,
    s1: Sequential<N, 10, Sigmoid>,
    s2: Sequential<10, Y, Softmax>,
}

impl<const N: usize, const X: usize, const Y: usize>
    NeuralNetwork<Y> for RnnSentimentAnalyzer<N, X, Y>
{
    type ModelInput = String;

    fn new(learn_rate: f64) -> Self {
        let embedding = Embedding::new("TODO");
        let rnn = RnnCell::new(learn_rate);
        let s1 = Sequential::new(learn_rate);
        let s2 = Sequential::new(learn_rate);
        Self {
            embedding,
            rnn,
            s1,
            s2,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f64, Y> {
        let x = self.embedding.embed(&x);
        let x = self.rnn.ff(x);
        let x = flatten(x);
        let x = self.s1.ff(x);
        let x = self.s2.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f64, Y>,
        y_test: SVector<f64, Y>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.s2.bp(g);
        let g = self.s1.bp(g);
        let g = unflatten(g);
        self.rnn.bp(g);
    }

    fn loss(
        y_out: &SVector<f64, Y>,
        y_test: &SVector<f64, Y>,
    ) -> f64 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten<const T: usize>(
    v: [SVector<f64, 1>; T],
) -> SVector<f64, T> {
    let mut out = SVector::zeros();
    for t in 0..T {
        out[t] = v[t][0];
    }
    out
}

fn unflatten<const T: usize>(
    v: SVector<f64, T>,
) -> [SVector<f64, 1>; T] {
    let mut out = [SVector::zeros(); T];
    for t in 0..T {
        out[t] = Vector1::new(v[t]);
    }
    out
}
