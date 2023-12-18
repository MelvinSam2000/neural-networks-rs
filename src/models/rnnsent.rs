use nalgebra::SVector;

use crate::activation::sigmoid::Sigmoid;
use crate::activation::tanh::Tanh;
use crate::layers::embedding::Embedding;
use crate::layers::rnncell::RnnCell;
use crate::layers::sequential::Sequential;

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
    s2: Sequential<10, Y, Sigmoid>,
}

impl<const N: usize, const X: usize, const Y: usize>
    RnnSentimentAnalyzer<N, X, Y>
{
    pub fn new() -> Self {
        todo!()
    }

    fn feedforward(
        &mut self,
        words: String,
    ) -> SVector<f64, Y> {
        let x = self.embedding.embed(&words);
        let x = self.rnn.ff(x);
        let x = Self::flatten(x);
        let x = self.s1.ff(x);
        let x = self.s2.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f64, Y>,
        y_test: SVector<f64, Y>,
    ) {
        let g = y_out - y_test;
        let g = self.s2.bp(g);
        let g = self.s1.bp(g);
        let g = Self::unflatten(g);
        self.rnn.bp(g);
    }

    fn flatten<const T: usize>(
        v: [SVector<f64, 1>; T],
    ) -> SVector<f64, T> {
        todo!()
    }

    fn unflatten<const T: usize>(
        v: SVector<f64, T>,
    ) -> [SVector<f64, 1>; T] {
        todo!()
    }
}
