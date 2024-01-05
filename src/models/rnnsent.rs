use nalgebra::SMatrix;
use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::sigmoid::Sigmoid;
use crate::layers::dense::Dense;
use crate::layers::rnncell::RnnCell;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::adam::AdamFactory;
use crate::optimizers::OptimizerFactory;

const H: usize = 100;

pub const HIDDEN_LAYER_DIM: usize = 10;
pub const HIDDEN_LAYER_NUM: usize = 1;
// N is number of words, X is dim of word embedding, Y is sentiment dimensions
pub struct RnnSentimentAnalyzer<
    const N: usize,
    const X: usize,
    const Y: usize,
    O: OptimizerFactory<HIDDEN_LAYER_DIM, H>
        + OptimizerFactory<HIDDEN_LAYER_DIM, 1>
        + OptimizerFactory<HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM>
        + OptimizerFactory<Y, HIDDEN_LAYER_DIM>
        + OptimizerFactory<Y, 1>,
> {
    rnn: RnnCell<
        X,
        H,
        H,
        N,
        AdamFactory<1, 100, 9, 10, 9, 10>,
    >,
    dense: Dense<
        H,
        Y,
        HIDDEN_LAYER_DIM,
        HIDDEN_LAYER_NUM,
        Sigmoid,
        O,
    >,
}

impl<const N: usize, const X: usize, const Y: usize, O>
    NeuralNetwork<Y> for RnnSentimentAnalyzer<N, X, Y, O>
where
    O: OptimizerFactory<HIDDEN_LAYER_DIM, H>
        + OptimizerFactory<HIDDEN_LAYER_DIM, 1>
        + OptimizerFactory<HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM>
        + OptimizerFactory<Y, HIDDEN_LAYER_DIM>
        + OptimizerFactory<Y, 1>,
{
    type ModelInput = SMatrix<f32, N, X>;

    fn new() -> Self {
        let rnn = RnnCell::new();
        let dense = Dense::new();
        Self { rnn, dense }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, Y> {
        let x = mat_to_array(x);
        let x = self.rnn.ff(x);
        let x = x[N - 1];
        let x = self.dense.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, Y>,
        y_test: SVector<f32, Y>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.dense.bp(g);
        let mut garr = [SMatrix::zeros(); N];
        garr[N - 1] = g;
        self.rnn.bp(garr);
    }

    fn loss(
        y_out: &SVector<f32, Y>,
        y_test: &SVector<f32, Y>,
    ) -> f32 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
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
