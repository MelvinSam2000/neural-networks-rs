use nalgebra::SMatrix;
use nalgebra::SVector;

use super::NeuralNetwork;
use crate::activation::relu::Relu;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::softmax::Softmax;
use crate::layers::actlayer::ActivationLayer;
use crate::layers::conv::Conv2d;
use crate::layers::maxpool::MaxPool2d;
use crate::layers::sequential::Sequential;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;

pub const MNIST_IMAGE_DIM: usize = 28;
const POST_CONV_DIM: usize = 20;
const CONV_WEIGHT_DIM: usize =
    MNIST_IMAGE_DIM - POST_CONV_DIM + 1;
const POST_CONV_DIM_SQUARED: usize =
    POST_CONV_DIM * POST_CONV_DIM;

const POST_POOL_DIM: usize = 15;
const POOL_FILTER_DIM: usize =
    POST_CONV_DIM - POST_POOL_DIM + 1;

const NUM_CONV: usize = 10;

const SEQ_LAYER_INITIAL_DIM: usize =
    POST_POOL_DIM * POST_POOL_DIM * NUM_CONV;

pub const DIGITS: usize = 10;

pub struct MyCnn {
    conv: Vec<
        Conv2d<
            MNIST_IMAGE_DIM,
            MNIST_IMAGE_DIM,
            POST_CONV_DIM,
            POST_CONV_DIM,
            CONV_WEIGHT_DIM,
            CONV_WEIGHT_DIM,
        >,
    >,
    relu: ActivationLayer<POST_CONV_DIM_SQUARED, Relu>,
    maxpool: Vec<
        MaxPool2d<
            POST_CONV_DIM,
            POST_CONV_DIM,
            POST_POOL_DIM,
            POST_POOL_DIM,
            POOL_FILTER_DIM,
            POOL_FILTER_DIM,
        >,
    >,
    s1: Sequential<SEQ_LAYER_INITIAL_DIM, 100, Sigmoid>,
    s2: Sequential<100, 50, Sigmoid>,
    s3: Sequential<50, 20, Sigmoid>,
    s4: Sequential<20, DIGITS, Softmax>,
}

impl NeuralNetwork<DIGITS> for MyCnn {
    type ModelInput =
        SMatrix<f64, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>;

    fn new(learn_rate: f64) -> Self {
        // initialize convolutional layers
        let conv = (0..NUM_CONV)
            .map(|_| Conv2d::new(learn_rate))
            .collect();

        let relu = ActivationLayer::new();

        let maxpool = (0..NUM_CONV)
            .map(|_| MaxPool2d::new())
            .collect();

        let s1 = Sequential::new(learn_rate);
        let s2 = Sequential::new(learn_rate);
        let s3 = Sequential::new(learn_rate);
        let s4 = Sequential::new(learn_rate);

        Self {
            conv,
            relu,
            maxpool,
            s1,
            s2,
            s3,
            s4,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f64, DIGITS> {
        let x = &x;
        let mut conv_results = [SMatrix::zeros(); NUM_CONV];
        for i in 0..NUM_CONV {
            let x = x.clone();
            let x = self.conv[i].ff(x);
            let x = mat_to_vec(x);
            let x = self.relu.ff(x);
            let x = vec_to_mat(x);
            let x = self.maxpool[i].ff(x);
            conv_results[i] = x;
        }
        let x = flatten(conv_results);
        let x = self.s1.ff(x);
        let x = self.s2.ff(x);
        let x = self.s3.ff(x);
        let x = self.s4.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f64, DIGITS>,
        y_test: SVector<f64, DIGITS>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.s4.bp(g);
        let g = self.s3.bp(g);
        let g = self.s2.bp(g);
        let g = self.s1.bp(g);
        let g = unflatten(g);
        for i in 0..NUM_CONV {
            let g = self.maxpool[i].bp(g[i]);
            let g = mat_to_vec(g);
            let g = self.relu.bp(g);
            let g = vec_to_mat(g);
            self.conv[i].bp(g);
        }
    }

    fn loss(
        y_out: &SVector<f64, DIGITS>,
        y_test: &SVector<f64, DIGITS>,
    ) -> f64 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten(
    v: [SMatrix<f64, POST_POOL_DIM, POST_POOL_DIM>;
        NUM_CONV],
) -> SVector<f64, SEQ_LAYER_INITIAL_DIM> {
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
    v: SVector<f64, SEQ_LAYER_INITIAL_DIM>,
) -> [SMatrix<f64, POST_POOL_DIM, POST_POOL_DIM>; NUM_CONV]
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

fn mat_to_vec(
    m: SMatrix<f64, POST_CONV_DIM, POST_CONV_DIM>,
) -> SVector<f64, POST_CONV_DIM_SQUARED> {
    let mut out = SVector::zeros();
    for i in 0..POST_CONV_DIM {
        for j in 0..POST_CONV_DIM {
            out[i * POST_CONV_DIM + j] = m[(i, j)];
        }
    }
    out
}

fn vec_to_mat(
    v: SVector<f64, POST_CONV_DIM_SQUARED>,
) -> SMatrix<f64, POST_CONV_DIM, POST_CONV_DIM> {
    let mut out = SMatrix::zeros();
    for i in 0..POST_CONV_DIM {
        for j in 0..POST_CONV_DIM {
            out[(i, j)] = v[i * POST_CONV_DIM + j];
        }
    }
    out
}
