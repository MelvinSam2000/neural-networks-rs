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
use mnist::MnistBuilder;
use models::cnn::DIGITS;
use models::cnn::MNIST_IMAGE_DIM;
use nalgebra::SMatrix;
use nalgebra::SVector;
use ndarray::Array2;
use ndarray::Array3;
use rand_distr::num_traits::Zero;

use crate::dataset::get_data_csv;
use crate::models::ann4::Ann4;
use crate::models::cnn::MyCnn;
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

fn write_costs_to_file(file: &str, recv: Receiver<f64>) {
    let mut costs = vec![];
    while let Ok(cost) = recv.recv() {
        costs.push(cost);
    }
    let mut destfile =
        File::create(&format!("debug/{file}.txt"))
            .expect("Please run 'mkdir debug'");
    for cost in costs {
        writeln!(destfile, "{}", cost).unwrap();
    }
}

fn main() {
    let cmd = std::env::args()
        .skip(1)
        .next()
        .expect("No CLI argument supplied");
    match cmd.as_str() {
        "ann" => train_and_validate_csv_ann(),
        "cnn" => train_and_validate_mnist_cnn(),
        _ => eprintln!("Invalid cmd provided"),
    };
}

fn train_and_validate_csv_ann() {
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

fn preprocess_narray_to_nalgebra(
    x_nd: Array3<f64>,
    y_nd: Array2<f64>,
) -> (
    Vec<SMatrix<f64, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>>,
    Vec<SVector<f64, DIGITS>>,
) {
    let (n, r, c) = x_nd.dim();
    let mut x_out = vec![];
    let mut y_out = vec![];
    for k in 0..n {
        let mut x = SMatrix::zeros();
        for i in 0..r {
            for j in 0..c {
                x[(i, j)] = x_nd[[k, i, j]];
            }
        }
        x_out.push(x);
        let mut y = SVector::zero();
        y[y_nd[[k, 0]] as usize % 10] = 1.0;
        y_out.push(y);
    }
    (x_out, y_out)
}

fn train_and_validate_mnist_cnn() {
    let (tx, rx) = mpsc::channel();

    let mut model =
        NNClassifierModel::<MyCnn, 10>::new(0.1, Some(tx));

    let dbg_thread = std::thread::spawn(move || {
        write_costs_to_file("cnn.txt", rx);
    });

    println!("CNN Model created");

    const TRAINING_DATASET: usize = 20000;
    const TEST_DATASET: usize = 2000;

    let mnist = Box::new(
        MnistBuilder::new()
            .download_and_extract()
            .label_format_digit()
            .training_set_length(TRAINING_DATASET as u32)
            .test_set_length(TEST_DATASET as u32)
            .finalize(),
    );

    let trn_img = mnist.trn_img;
    let trn_lbl = mnist.trn_lbl;
    let tst_img = mnist.tst_img;
    let tst_lbl = mnist.tst_lbl;

    println!("Loaded MNIST data");

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec(
        (
            TRAINING_DATASET,
            MNIST_IMAGE_DIM,
            MNIST_IMAGE_DIM,
        ),
        trn_img,
    )
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec(
        (TRAINING_DATASET, 1),
        trn_lbl,
    )
    .expect(
        "Error converting training labels to Array2 struct",
    )
    .map(|x| *x as f64);

    let test_data: Array3<f64> = Array3::from_shape_vec(
        (TEST_DATASET, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM),
        tst_img,
    )
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.);

    let test_labels: Array2<f64> =
        Array2::from_shape_vec((TEST_DATASET, 1), tst_lbl)
            .expect(
                "Error converting testing labels to \
                 Array2 struct",
            )
            .map(|x| *x as f64);

    let (x_train, y_train) = preprocess_narray_to_nalgebra(
        train_data,
        train_labels,
    );
    let (x_test, y_test) = preprocess_narray_to_nalgebra(
        test_data,
        test_labels,
    );

    let y_test = y_test
        .into_iter()
        .map(|v| {
            for i in 0..DIGITS {
                if v[i] != 0. {
                    return i;
                }
            }
            unreachable!()
        })
        .collect::<Vec<usize>>();

    println!("Training began...");
    model.train(&x_train, &y_train);
    let score = model.validate(&x_test, &y_test);
    //let score = y_train.len() as f64;

    println!("CNN Score: {:.3}%\t", score * 100.);
    drop(model);
    dbg_thread.join().unwrap();
}
