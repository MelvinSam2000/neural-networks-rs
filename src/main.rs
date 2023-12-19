#![feature(maybe_uninit_uninit_array, maybe_uninit_slice)]

use std::fs::File;
use std::io::Write;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use activation::relu::Relu;
use activation::sigmoid::Sigmoid;
use activation::softmax::Softmax;
use activation::ActivationFunction;
use loss::crossent::CrossEntropy;
use loss::mse::Mse;
use loss::LossFunction;

use crate::dataset::get_data_csv;
use crate::models::ann4::Ann4;
use crate::models::NNClassifierModel;

pub mod activation;
pub mod dataset;
pub mod layers;
pub mod loss;
pub mod models;

fn train_and_validate<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
    F1: ActivationFunction<L2>,
    F2: ActivationFunction<L3>,
    F3: ActivationFunction<L4>,
    LOSS: LossFunction<L4>,
>(
    learn_rate: f64,
    csv_file: &str,
    debug_channel: Option<Sender<f64>>,
) {
    let (x_train, y_train, x_test, y_test) =
        get_data_csv(csv_file, 0.8)
            .expect("Could not read data from csv file");

    let (x_train, y_train) = Ann4::<
        L1,
        L2,
        L3,
        L4,
        F1,
        F2,
        F3,
        LOSS,
    >::preprocess(
        &x_train, &y_train
    );
    let (x_test, _) = Ann4::<
        L1,
        L2,
        L3,
        L4,
        F1,
        F2,
        F3,
        LOSS,
    >::preprocess(&x_test, &y_test);
    let mut model = NNClassifierModel::<
        Ann4<L1, L2, L3, L4, F1, F2, F3, LOSS>,
        L4,
    >::new(learn_rate, debug_channel);
    model.train(&x_train, &y_train);
    let score = model.validate(&x_test, &y_test);

    println!(
        "File: {}\t LR:{}\t Score: {:.3}%\t",
        csv_file,
        learn_rate,
        score * 100.
    );
}

fn write_costs_to_file(
    csv_file: &str,
    recv: Receiver<f64>,
) {
    let mut costs = vec![];
    while let Ok(cost) = recv.recv() {
        costs.push(cost);
    }
    let mut destfile =
        File::create(&format!("debug/{csv_file}"))
            .expect("Please run 'mkdir debug'");
    for cost in costs {
        writeln!(destfile, "{}", cost).unwrap();
    }
}

fn main() {
    let tasks = vec![
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                4,
                6,
                3,
                Sigmoid,
                Sigmoid,
                Softmax,
                CrossEntropy,
            >(0.8, "data/knn.csv", Some(tx));
            write_costs_to_file("knn.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                7,
                10,
                3,
                Relu,
                Relu,
                Softmax,
                CrossEntropy,
            >(0.001, "data/gda.csv", Some(tx));
            write_costs_to_file("gda.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                8,
                10,
                6,
                2,
                Relu,
                Relu,
                Sigmoid,
                Mse,
            >(0.001, "data/nb.csv", Some(tx));
            write_costs_to_file("nb.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                12,
                14,
                2,
                Sigmoid,
                Sigmoid,
                Softmax,
                CrossEntropy,
            >(
                0.5, "data/neg_square.csv", Some(tx)
            );
            write_costs_to_file("neg_square.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                15,
                13,
                2,
                Relu,
                Sigmoid,
                Softmax,
                CrossEntropy,
            >(0.5, "data/circle.csv", Some(tx));
            write_costs_to_file("circle.csv", rx);
        },
    ];
    tasks
        .into_iter()
        .map(|task| std::thread::spawn(task))
        .for_each(|task| {
            task.join().expect("A task failed")
        });
}
