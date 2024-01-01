use nalgebra::SMatrix;
use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::relu::Relu;
use crate::activation::sigmoid::Sigmoid;
use crate::layers::attention::Attention;
use crate::layers::posencoder::PosEncoder;
use crate::layers::sequential::Sequential;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::OptimizerFactory;

const N: usize = 50;
const M: usize = 200;
const Y: usize = 2;
const D: usize = 10;
const NM: usize = N * M;
const T: usize = 2;
const L1: usize = 100;
const L2: usize = 50;
const L3: usize = 10;

pub struct Transformer1<
    O: OptimizerFactory<M, D>
        + OptimizerFactory<M, M>
        + OptimizerFactory<NM, NM>
        + OptimizerFactory<NM, 1>
        + OptimizerFactory<L1, NM>
        + OptimizerFactory<L1, 1>
        + OptimizerFactory<L2, L1>
        + OptimizerFactory<L2, 1>
        + OptimizerFactory<L3, L2>
        + OptimizerFactory<L3, 1>
        + OptimizerFactory<Y, L3>
        + OptimizerFactory<Y, 1>,
> {
    posencoder: PosEncoder<N, M>,
    attention: [Attention<M, N, D, O>; T],
    seq: [Sequential<NM, NM, Relu, O>; T],
    seqf1: Sequential<NM, L1, Sigmoid, O>,
    seqf2: Sequential<L1, L2, Sigmoid, O>,
    seqf3: Sequential<L2, L3, Sigmoid, O>,
    seqf4: Sequential<L3, Y, Sigmoid, O>,
}

impl<O> NeuralNetwork<Y> for Transformer1<O>
where
    O: OptimizerFactory<M, D>
        + OptimizerFactory<M, M>
        + OptimizerFactory<NM, NM>
        + OptimizerFactory<NM, 1>
        + OptimizerFactory<L1, NM>
        + OptimizerFactory<L1, 1>
        + OptimizerFactory<L2, L1>
        + OptimizerFactory<L2, 1>
        + OptimizerFactory<L3, L2>
        + OptimizerFactory<L3, 1>
        + OptimizerFactory<Y, L3>
        + OptimizerFactory<Y, 1>,
{
    type ModelInput = SMatrix<f32, N, M>;

    fn new() -> Self {
        let posencoder = PosEncoder::new();

        #[rustfmt::skip]
        let attention = [
            Attention::new(),
            Attention::new(),
            //Attention::new(),
        ];

        #[rustfmt::skip]
        let seq = [
            Sequential::new(),
            Sequential::new(),
            //Sequential::new(),
        ];

        let seqf1 = Sequential::new();
        let seqf2 = Sequential::new();
        let seqf3 = Sequential::new();
        let seqf4 = Sequential::new();

        Self {
            posencoder,
            attention,
            seq,
            seqf1,
            seqf2,
            seqf3,
            seqf4,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, Y> {
        let mut x = self.posencoder.ff(x);
        for t in 0..T {
            x = self.attention[t].ff(x);
            let x_tmp = flatten(x);
            let x_tmp = self.seq[t].ff(x_tmp);
            x = unflatten(x_tmp);
        }
        let x = flatten(x);
        let x = self.seqf1.ff(x);
        let x = self.seqf2.ff(x);
        let x = self.seqf3.ff(x);
        let x = self.seqf4.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, Y>,
        y_test: SVector<f32, Y>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.seqf4.bp(g);
        let g = self.seqf3.bp(g);
        let g = self.seqf2.bp(g);
        let g = self.seqf1.bp(g);
        let mut g = unflatten(g);
        for t in (0..T).rev() {
            let g_tmp = flatten(g);
            let g_tmp = self.seq[t].bp(g_tmp);
            let g_tmp = unflatten(g_tmp);
            g = self.attention[t].bp(g_tmp);
        }
    }

    fn loss(
        y_out: &nalgebra::SVector<f32, Y>,
        y_test: &nalgebra::SVector<f32, Y>,
    ) -> f32 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten(v: SMatrix<f32, N, M>) -> SVector<f32, NM> {
    let mut out = SVector::zeros();
    for i in 0..N {
        for j in 0..M {
            out[i * M + j] = v[(i, j)];
        }
    }
    out
}

fn unflatten(v: SVector<f32, NM>) -> SMatrix<f32, N, M> {
    let mut out = SMatrix::zeros();
    for i in 0..N {
        for j in 0..M {
            out[(i, j)] = v[i * M + j];
        }
    }
    out
}
