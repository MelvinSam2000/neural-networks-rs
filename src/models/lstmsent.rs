use nalgebra::SMatrix;
use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::relu::Relu;
use crate::layers::dense::Dense;
use crate::layers::lstm::Lstm;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::OptimizerFactory;

const N: usize = 50;
const M: usize = 200;
const MM: usize = 2 * M;
const L: usize = 40;

pub struct LstmSentAnalyzer<
    O: OptimizerFactory<L, M>
        + OptimizerFactory<L, 1>
        + OptimizerFactory<L, L>
        + OptimizerFactory<2, L>
        + OptimizerFactory<2, 1>
        + OptimizerFactory<M, MM>
        + OptimizerFactory<M, 1>,
> {
    lstm: Lstm<M, M, N, MM, O>,
    dense: Dense<M, 2, L, 5, Relu, O>,
}

impl<O> NeuralNetwork<2> for LstmSentAnalyzer<O>
where
    O: OptimizerFactory<L, M>
        + OptimizerFactory<L, 1>
        + OptimizerFactory<L, L>
        + OptimizerFactory<2, L>
        + OptimizerFactory<2, 1>
        + OptimizerFactory<M, MM>
        + OptimizerFactory<M, 1>,
{
    type ModelInput = SMatrix<f32, N, M>;

    fn new() -> Self {
        let lstm = Lstm::new();
        let dense = Dense::new();
        Self { lstm, dense }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, 2> {
        let x = mat_to_array(x);
        let x = self.lstm.ff(x);
        let x = x[N - 1];
        let x = self.dense.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, 2>,
        y_test: SVector<f32, 2>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g0 = self.dense.bp(g);
        let mut g = [SMatrix::zeros(); N];
        g[N - 1] = g0;
        self.lstm.bp(g);
    }

    fn loss(
        y_out: &SVector<f32, 2>,
        y_test: &SVector<f32, 2>,
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
