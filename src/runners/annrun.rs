use std::sync::mpsc::Sender;
use std::sync::mpsc::{self};

use super::write_costs_to_file;
use crate::activation::relu::Relu;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::ActivationFunction;
use crate::dataset::get_data_csv;
use crate::loss::crossent::CrossEntropy;
use crate::loss::LossFunction;
use crate::models::ann::Ann;
use crate::models::ann4::Ann4;
use crate::models::NNClassifierModel;
use crate::optimizers::adam::AdamFactory;
use crate::optimizers::rmsprop::RmsPropFactory;
use crate::optimizers::sgdmomentum::SgdWMomentumFactory;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

fn train_and_validate<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
    F1: ActivationFunction,
    F2: ActivationFunction,
    LOSS: LossFunction<L4>,
    OPT: OptimizerFactory<L2, L1>
        + OptimizerFactory<L2, L2>
        + OptimizerFactory<L4, L2>
        + OptimizerFactory<L2, 1>
        + OptimizerFactory<L3, L2>
        + OptimizerFactory<L3, 1>
        + OptimizerFactory<L4, L3>
        + OptimizerFactory<L4, 1>,
>(
    csv_file: &str,
    debug_channel: Option<Sender<f32>>,
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
        LOSS,
        OPT,
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
        LOSS,
        OPT,
    >::preprocess(&x_test, &y_test);
    let mut model = NNClassifierModel::<
        //Ann4<L1, L2, L3, L4, F1, F2, LOSS, OPT>,
        Ann<L1, L4, L2, 5, F1, LOSS, OPT>,
        L4,
    >::new(debug_channel);
    model.train(&x_train, &y_train);
    let score = model.validate(&x_test, &y_test);

    println!(
        "File: {}\t LR:{}\t Score: {:.3}%\t",
        csv_file,
        <OPT as OptimizerFactory<L2, 1>>::Optimizer::name(),
        score * 100.
    );
}

pub fn train_and_validate_csv_ann() {
    let tasks = vec![
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                10,
                6,
                3,
                Relu,
                Sigmoid,
                CrossEntropy,
                //SgdFactory<8, 10>,
                //RmsPropFactory<8, 10, 9, 10>,
                AdamFactory<1, 100, 9, 10, 9, 10>,
            >("data/knn.csv", Some(tx));
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
                Sigmoid,
                CrossEntropy,
                //SgdWMomentumFactory<1, 10, 8, 10>,
                //RmsPropFactory<1, 10, 9, 10>,
                AdamFactory<1, 100, 9, 10, 9, 10>,
            >("data/gda.csv", Some(tx));
            write_costs_to_file("gda.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                8,
                10,
                6,
                2,
                Sigmoid,
                Sigmoid,
                CrossEntropy,
                //SgdWMomentumFactory<1, 10, 5, 10>,
                //SgdFactory<1, 1000>,
                RmsPropFactory<1, 1000, 9, 10>,
                //AdamFactory<1, 1000, 8, 10, 8, 10>,
            >("data/nb.csv", Some(tx));
            write_costs_to_file("nb.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<
                2,
                12,
                14,
                2,
                Relu,
                Sigmoid,
                CrossEntropy,
                //SgdFactory<1, 2>,
                //RmsPropFactory<1, 2, 9, 10>,
                AdamFactory<1, 2, 8, 10, 8, 10>,
            >("data/neg_square.csv", Some(tx));
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
                CrossEntropy,
                //SgdFactory<1, 2>,
                SgdWMomentumFactory<1, 2, 8, 10>,
                //AdamFactory<1, 2, 8, 10, 8, 10>,
            >("data/circle.csv", Some(tx));
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
