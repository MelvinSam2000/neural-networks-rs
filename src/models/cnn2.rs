use nalgebra::SMatrix;
use nalgebra::SVector;

use super::cnn::DIGITS;
use super::cnn::MNIST_IMAGE_DIM;
use super::NeuralNetwork;
use crate::activation::noact::NoActivation;
use crate::activation::sigmoid::Sigmoid;
use crate::layers::conv::Conv2d;
use crate::layers::maxpool::MaxPool2d;
use crate::layers::relu2d::Relu2dLayer;
use crate::layers::sequential::Sequential;
use crate::layers::softmax::Softmax;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::optimizers::OptimizerFactory;

const POST_CONV1_DIM: usize = 20;
const CONV1_WEIGHT_DIM: usize =
    MNIST_IMAGE_DIM - POST_CONV1_DIM + 1;

const POST_POOL1_DIM: usize = 15;
const POOL1_FILTER_DIM: usize =
    POST_CONV1_DIM - POST_POOL1_DIM + 1;

const POST_CONV2_DIM: usize = 10;
const CONV2_WEIGHT_DIM: usize =
    POST_POOL1_DIM - POST_CONV2_DIM + 1;

const POST_POOL2_DIM: usize = 8;
const POOL2_FILTER_DIM: usize =
    POST_CONV2_DIM - POST_POOL2_DIM + 1;

const SEQ_LAYER_INITIAL_DIM: usize =
    POST_POOL2_DIM * POST_POOL2_DIM;

pub struct MyCnn2<
    OPT: OptimizerFactory<100, SEQ_LAYER_INITIAL_DIM>
        + OptimizerFactory<100, 1>
        + OptimizerFactory<50, 100>
        + OptimizerFactory<50, 1>
        + OptimizerFactory<20, 50>
        + OptimizerFactory<20, 1>
        + OptimizerFactory<DIGITS, 20>
        + OptimizerFactory<DIGITS, 1>
        + OptimizerFactory<CONV1_WEIGHT_DIM, CONV1_WEIGHT_DIM>
        + OptimizerFactory<CONV2_WEIGHT_DIM, CONV2_WEIGHT_DIM>,
> {
    conv1: Conv2d<
        MNIST_IMAGE_DIM,
        MNIST_IMAGE_DIM,
        POST_CONV1_DIM,
        POST_CONV1_DIM,
        CONV1_WEIGHT_DIM,
        CONV1_WEIGHT_DIM,
        OPT,
    >,
    relu1: Relu2dLayer<POST_CONV1_DIM, POST_CONV1_DIM>,
    maxpool1: MaxPool2d<
        POST_CONV1_DIM,
        POST_CONV1_DIM,
        POST_POOL1_DIM,
        POST_POOL1_DIM,
        POOL1_FILTER_DIM,
        POOL1_FILTER_DIM,
    >,
    conv2: Conv2d<
        POST_POOL1_DIM,
        POST_POOL1_DIM,
        POST_CONV2_DIM,
        POST_CONV2_DIM,
        CONV2_WEIGHT_DIM,
        CONV2_WEIGHT_DIM,
        OPT,
    >,
    relu2: Relu2dLayer<POST_CONV2_DIM, POST_CONV2_DIM>,
    maxpool2: MaxPool2d<
        POST_CONV2_DIM,
        POST_CONV2_DIM,
        POST_POOL2_DIM,
        POST_POOL2_DIM,
        POOL2_FILTER_DIM,
        POOL2_FILTER_DIM,
    >,
    s1: Sequential<
        SEQ_LAYER_INITIAL_DIM,
        100,
        Sigmoid,
        OPT,
    >,
    s2: Sequential<100, 50, Sigmoid, OPT>,
    s3: Sequential<50, 20, Sigmoid, OPT>,
    s4: Sequential<20, DIGITS, NoActivation, OPT>,
    softmax: Softmax<DIGITS>,
}

impl<OPT> NeuralNetwork<DIGITS> for MyCnn2<OPT>
where
    OPT: OptimizerFactory<100, SEQ_LAYER_INITIAL_DIM>
        + OptimizerFactory<100, 1>
        + OptimizerFactory<50, 100>
        + OptimizerFactory<50, 1>
        + OptimizerFactory<20, 50>
        + OptimizerFactory<20, 1>
        + OptimizerFactory<DIGITS, 20>
        + OptimizerFactory<DIGITS, 1>
        + OptimizerFactory<CONV1_WEIGHT_DIM, CONV1_WEIGHT_DIM>
        + OptimizerFactory<CONV2_WEIGHT_DIM, CONV2_WEIGHT_DIM>,
{
    type ModelInput =
        SMatrix<f32, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>;

    fn new() -> Self {
        // initialize convolutional layers
        let conv1 = Conv2d::new();
        let conv2 = Conv2d::new();

        let relu1 = Relu2dLayer::new();
        let relu2 = Relu2dLayer::new();

        let maxpool1 = MaxPool2d::new();
        let maxpool2 = MaxPool2d::new();

        let s1 = Sequential::new();
        let s2 = Sequential::new();
        let s3 = Sequential::new();
        let s4 = Sequential::new();

        let softmax = Softmax::new();

        Self {
            conv1,
            relu1,
            maxpool1,
            conv2,
            relu2,
            maxpool2,
            s1,
            s2,
            s3,
            s4,
            softmax,
        }
    }

    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f32, DIGITS> {
        let x = self.conv1.ff(x);
        let x = self.relu1.ff(x);
        let x = self.maxpool1.ff(x);
        let x = self.conv2.ff(x);
        let x = self.relu2.ff(x);
        let x = self.maxpool2.ff(x);
        let x = flatten(x);
        let x = self.s1.ff(x);
        let x = self.s2.ff(x);
        let x = self.s3.ff(x);
        let x = self.s4.ff(x);
        let x = self.softmax.ff(x);
        x
    }

    fn backprop(
        &mut self,
        y_out: SVector<f32, DIGITS>,
        y_test: SVector<f32, DIGITS>,
    ) {
        let g = CrossEntropy::grad(y_out, y_test);
        let g = self.softmax.bp(g);
        let g = self.s4.bp(g);
        let g = self.s3.bp(g);
        let g = self.s2.bp(g);
        let g = self.s1.bp(g);
        let g = unflatten(g);
        let g = self.maxpool2.bp(g);
        let g = self.relu2.bp(g);
        let g = self.conv2.bp(g);
        let g = self.maxpool1.bp(g);
        let g = self.relu1.bp(g);
        self.conv1.bp(g);
    }

    fn loss(
        y_out: &SVector<f32, DIGITS>,
        y_test: &SVector<f32, DIGITS>,
    ) -> f32 {
        CrossEntropy::func(y_out.clone(), y_test.clone())
    }
}

fn flatten(
    v: SMatrix<f32, POST_POOL2_DIM, POST_POOL2_DIM>,
) -> SVector<f32, SEQ_LAYER_INITIAL_DIM> {
    let mut out = SVector::zeros();
    for i in 0..POST_POOL2_DIM {
        for j in 0..POST_POOL2_DIM {
            out[i * POST_POOL2_DIM + j] = v[(i, j)];
        }
    }
    out
}

fn unflatten(
    v: SVector<f32, SEQ_LAYER_INITIAL_DIM>,
) -> SMatrix<f32, POST_POOL2_DIM, POST_POOL2_DIM> {
    let mut out = SMatrix::zeros();
    for i in 0..POST_POOL2_DIM {
        for j in 0..POST_POOL2_DIM {
            out[(i, j)] = v[i * POST_POOL2_DIM + j];
        }
    }
    out
}
