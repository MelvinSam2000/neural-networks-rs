use std::sync::mpsc::Sender;

use nalgebra::SVector;

use crate::activation::ActivationFunction;
use crate::layers::sequential::Sequential;

pub struct Ann4<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
    F1,
    F2,
    F3,
> {
    s1: Sequential<L1, L2, F1>,
    s2: Sequential<L2, L3, F2>,
    s3: Sequential<L3, L4, F3>,
    debug_channel: Option<Sender<f64>>,
}

impl<
        const L1: usize,
        const L2: usize,
        const L3: usize,
        const L4: usize,
        F1,
        F2,
        F3,
    > Ann4<L1, L2, L3, L4, F1, F2, F3>
where
    F1: ActivationFunction<L2>,
    F2: ActivationFunction<L3>,
    F3: ActivationFunction<L4>,
{
    pub fn new(
        learn_rate: f64,
        debug_channel: Option<Sender<f64>>,
    ) -> Self {
        let s1 = Sequential::new(learn_rate);
        let s2 = Sequential::new(learn_rate);
        let s3 = Sequential::new(learn_rate);
        Self {
            s1,
            s2,
            s3,
            debug_channel,
        }
    }

    pub fn feedforward(
        &mut self,
        x: SVector<f64, L1>,
    ) -> SVector<f64, L4> {
        let a = x;
        let a = self.s1.ff(a);
        let a = self.s2.ff(a);
        let a = self.s3.ff(a);
        a
    }

    pub fn backprop(
        &mut self,
        y_output: SVector<f64, L4>,
        y_test: SVector<f64, L4>,
    ) {
        let g = y_output - y_test;
        let g = self.s3.bp(g);
        let g = self.s2.bp(g);
        self.s1.bp(g);
    }

    fn preprocess(
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

    pub fn train(
        &mut self,
        x_train: &[[f64; L1]],
        y_train: &[usize],
    ) {
        if x_train.len() != y_train.len() {
            panic!(
                "x_train and y_train have different sizes \
                 of samples"
            );
        }

        let (x_train, y_train) =
            Self::preprocess(x_train, y_train);
        // begin training
        let n = x_train.len();
        let k = n / 100;
        for i in 0..n {
            let x = x_train[i];
            let y = y_train[i];
            let y_out = self.feedforward(x);
            if let Some(channel) =
                self.debug_channel.as_ref()
            {
                if i % k == 0 {
                    let cost = (y - y_out).norm_squared();
                    channel.send(cost).unwrap();
                }
            }
            self.backprop(y_out, y);
        }
    }

    pub fn predict(
        &mut self,
        x: &SVector<f64, L1>,
    ) -> usize {
        let y = self.feedforward(*x);
        y.iter()
            .enumerate()
            .reduce(|(max_y, max_prob), (y, prob)| {
                if prob > max_prob {
                    (y, prob)
                } else {
                    (max_y, max_prob)
                }
            })
            .unwrap()
            .0
    }

    pub fn validate(
        &mut self,
        x_test: &[[f64; L1]],
        y_test: &[usize],
    ) -> f64 {
        if x_test.len() != y_test.len() {
            panic!(
                "x_test and y_test have different sizes \
                 of samples"
            );
        }

        let x_test: Vec<SVector<f64, L1>> = x_test
            .iter()
            .map(|x| SVector::from_column_slice(x))
            .collect();

        let n = x_test.len();
        let count: f64 = (0..n)
            .map(|i| {
                if self.predict(&x_test[i]) == y_test[i] {
                    1.
                } else {
                    0.
                }
            })
            .sum();
        count / n as f64
    }
}
