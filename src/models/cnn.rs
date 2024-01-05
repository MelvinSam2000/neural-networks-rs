use nalgebra::SMatrix;
use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::relu::Relu;
use crate::layers::conv::Conv2d;
use crate::layers::dense::Dense;
use crate::layers::maxpool::MaxPool2d;
use crate::layers::relu2d::Relu2dLayer;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::OptimizerFactory;

pub const MNIST_IMAGE_DIM: usize = 28;
const POST_CONV_DIM: usize = 20;
const CONV_WEIGHT_DIM: usize =
    MNIST_IMAGE_DIM - POST_CONV_DIM + 1;

const POST_POOL_DIM: usize = 15;
const POOL_FILTER_DIM: usize =
    POST_CONV_DIM - POST_POOL_DIM + 1;

const NUM_CONV: usize = 4;

const SEQ_LAYER_INITIAL_DIM: usize =
    POST_POOL_DIM * POST_POOL_DIM * NUM_CONV;

pub const DIGITS: usize = 10;

pub const HIDDEN_LAYER_DIM: usize = 100;
pub const HIDDEN_LAYER_NUM: usize = 2;

pub struct MyCnn<
    OPT: OptimizerFactory<CONV_WEIGHT_DIM, CONV_WEIGHT_DIM>
        + OptimizerFactory<
            HIDDEN_LAYER_DIM,
            SEQ_LAYER_INITIAL_DIM,
        > + OptimizerFactory<HIDDEN_LAYER_DIM, 1>
        + OptimizerFactory<HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM>
        + OptimizerFactory<DIGITS, HIDDEN_LAYER_DIM>
        + OptimizerFactory<DIGITS, 1>,
> {
    conv: [Conv2d<
        MNIST_IMAGE_DIM,
        MNIST_IMAGE_DIM,
        POST_CONV_DIM,
        POST_CONV_DIM,
        CONV_WEIGHT_DIM,
        CONV_WEIGHT_DIM,
        OPT,
    >; NUM_CONV],
    relu: Relu2dLayer<POST_CONV_DIM, POST_CONV_DIM>,
    maxpool: [MaxPool2d<
        POST_CONV_DIM,
        POST_CONV_DIM,
        POST_POOL_DIM,
        POST_POOL_DIM,
        POOL_FILTER_DIM,
        POOL_FILTER_DIM,
    >; NUM_CONV],
    dense: Dense<
        SEQ_LAYER_INITIAL_DIM,
        DIGITS,
        HIDDEN_LAYER_DIM,
        HIDDEN_LAYER_NUM,
        Relu,
        OPT,
    >,
}

impl<OPT> NeuralNetwork<DIGITS> for MyCnn<OPT>
where
    OPT: OptimizerFactory<CONV_WEIGHT_DIM, CONV_WEIGHT_DIM>
        + OptimizerFactory<
            HIDDEN_LAYER_DIM,
            SEQ_LAYER_INITIAL_DIM,
        > + OptimizerFactory<HIDDEN_LAYER_DIM, 1>
        + OptimizerFactory<HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM>
        + OptimizerFactory<DIGITS, HIDDEN_LAYER_DIM>
        + OptimizerFactory<DIGITS, 1>,
{
    type ModelInput =
        SMatrix<f32, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>;

    fn new() -> Self {
        // initialize convolutional layers
        let conv = [
            Conv2d::new(),
            Conv2d::new(),
            Conv2d::new(),
            Conv2d::new(),
        ];

        let relu = Relu2dLayer::new();

        let maxpool = [MaxPool2d::new(); NUM_CONV];

        let dense = Dense::new();

        Self {
            conv,
            relu,
            maxpool,
            dense,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, DIGITS> {
        let x = &x;
        let mut conv_results = [SMatrix::zeros(); NUM_CONV];
        for i in 0..NUM_CONV {
            let x = x.clone();
            let x = self.conv[i].ff(x);
            let x = self.relu.ff(x);
            let x = self.maxpool[i].ff(x);
            conv_results[i] = x;
        }
        let x = flatten(conv_results);
        let x = self.dense.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, DIGITS>,
        y_test: SVector<f32, DIGITS>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.dense.bp(g);
        let g = unflatten(g);
        for i in 0..NUM_CONV {
            let g = self.maxpool[i].bp(g[i]);
            let g = self.relu.bp(g);
            self.conv[i].bp(g);
        }
    }

    fn loss(
        y_out: &SVector<f32, DIGITS>,
        y_test: &SVector<f32, DIGITS>,
    ) -> f32 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten(
    v: [SMatrix<f32, POST_POOL_DIM, POST_POOL_DIM>;
        NUM_CONV],
) -> SVector<f32, SEQ_LAYER_INITIAL_DIM> {
    let mut out = SVector::zeros();
    for k in 0..NUM_CONV {
        for i in 0..POST_POOL_DIM {
            for j in 0..POST_POOL_DIM {
                out[k * POST_POOL_DIM * POST_POOL_DIM
                    + i * POST_POOL_DIM
                    + j] = v[k][(i, j)];
            }
        }
    }
    out
}

fn unflatten(
    v: SVector<f32, SEQ_LAYER_INITIAL_DIM>,
) -> [SMatrix<f32, POST_POOL_DIM, POST_POOL_DIM>; NUM_CONV]
{
    let mut out = [SMatrix::zeros(); NUM_CONV];
    for k in 0..NUM_CONV {
        for i in 0..POST_POOL_DIM {
            for j in 0..POST_POOL_DIM {
                out[k][(i, j)] =
                    v[k * POST_POOL_DIM * POST_POOL_DIM
                        + i * POST_POOL_DIM
                        + j];
            }
        }
    }
    out
}
