use std::sync::mpsc::Sender;

use nalgebra::SVector;

pub mod ann4;
pub mod cnn;
pub mod rnnsent;

pub trait NeuralNetwork<const Y: usize> {
    type ModelInput;
    fn new(learn_rate: f64) -> Self;
    fn feedforward(
        &mut self,
        x: Self::ModelInput,
    ) -> SVector<f64, Y>;
    fn backprop(
        &mut self,
        y_out: SVector<f64, Y>,
        y_test: SVector<f64, Y>,
    );
    fn loss(
        y_out: &SVector<f64, Y>,
        y_test: &SVector<f64, Y>,
    ) -> f64;
}

pub struct NNClassifierModel<T, const Y: usize> {
    model: T,
    debug_channel: Option<Sender<f64>>,
}

impl<T, const Y: usize> NNClassifierModel<T, Y>
where
    T: NeuralNetwork<Y>,
    T::ModelInput: Clone,
    T::ModelInput: Copy,
{
    pub fn new(
        learn_rate: f64,
        debug_channel: Option<Sender<f64>>,
    ) -> Self {
        let model = T::new(learn_rate);
        Self {
            model,
            debug_channel,
        }
    }

    pub fn train(
        &mut self,
        x_train: &[T::ModelInput],
        y_train: &[SVector<f64, Y>],
    ) {
        if x_train.len() != y_train.len() {
            panic!(
                "x_train and y_train have different sizes \
                 of samples"
            );
        }
        // begin training
        let n = x_train.len();
        const M: usize = 200;
        let k = n / M;
        for i in 0..n {
            let x = x_train[i];
            let y = y_train[i];
            let y_out = self.model.feedforward(x);

            if let Some(channel) =
                self.debug_channel.as_ref()
            {
                if n < M || i % k == 0 {
                    print!(
                        "Training completion: \r{:.0}%",
                        (i as f64 / n as f64) * 100.
                    );
                    let cost = T::loss(&y_out, &y);
                    channel.send(cost).unwrap();
                }
            }
            self.model.backprop(y_out, y);
        }
    }

    pub fn predict(&mut self, x: T::ModelInput) -> usize {
        let y = self.model.feedforward(x);
        y.into_iter()
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
        x_test: &[T::ModelInput],
        y_test: &[usize],
    ) -> f64 {
        if x_test.len() != y_test.len() {
            panic!(
                "x_test and y_test have different sizes \
                 of samples"
            );
        }

        let n = x_test.len();
        let count: f64 = (0..n)
            .map(|i| {
                if self.predict(x_test[i]) == y_test[i] {
                    1.
                } else {
                    0.
                }
            })
            .sum();
        count / n as f64
    }
}
