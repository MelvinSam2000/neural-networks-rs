use std::sync::mpsc;

use mnist::MnistBuilder;
use nalgebra::SMatrix;
use nalgebra::SVector;
use ndarray::Array2;
use ndarray::Array3;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPoolBuilder;

use crate::models::cnn::MyCnn;
use crate::models::cnn::DIGITS;
use crate::models::cnn::MNIST_IMAGE_DIM;
use crate::models::NNClassifierModel;
use crate::optimizers::adagrad::AdagradFactory;
use crate::runners::write_costs_to_file;

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
        let mut y = SVector::zeros();
        y[y_nd[[k, 0]] as usize % 10] = 1.0;
        y_out.push(y);
    }
    (x_out, y_out)
}

fn get_datasets() -> (
    Vec<SMatrix<f64, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>>,
    Vec<SVector<f64, DIGITS>>,
    Vec<SMatrix<f64, MNIST_IMAGE_DIM, MNIST_IMAGE_DIM>>,
    Vec<SVector<f64, DIGITS>>,
) {
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

    (x_train, y_train, x_test, y_test)
}

pub fn train_and_validate_mnist_cnn() {
    let pool = ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build()
        .unwrap();

    let (x_train, y_train, x_test, y_test) = get_datasets();

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

    pool.install(|| {
        (0..6)
            .into_par_iter()
            .map(|i| {
                let (tx, rx) = mpsc::channel();
                let mut model =
                    NNClassifierModel::<
                        MyCnn<
                            //SgdWMomentumFactory<1, 100, 5, 10>,
                            //SgdFactory<1, 10>,
                            //RmsPropFactory<1, 100, 9, 10>,
                            AdagradFactory<1, 100>,
                        >,
                        10,
                    >::new(Some(tx));
                let dbg_thread =
                    std::thread::spawn(move || {
                        write_costs_to_file(
                            &format!("cnn-{i}.txt"),
                            rx,
                        );
                    });
                model.train(&x_train, &y_train);
                println!("");
                (
                    i,
                    model.validate(&x_test, &y_test),
                    dbg_thread,
                )
            })
            .for_each(|(id, score, dbg_thread)| {
                println!(
                    "CNN Score thread-{id}: {:.3}%\t",
                    score * 100.
                );
                dbg_thread.join().unwrap();
            });
    });
}
