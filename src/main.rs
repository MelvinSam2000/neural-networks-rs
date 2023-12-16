use std::fs::File;
use std::io::Write;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use crate::ann4::Ann4;
use crate::dataset::get_data_csv;

pub mod activation;
pub mod ann4;
pub mod dataset;
pub mod rnncell;
pub mod sequential;

fn train_and_validate<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
>(
    learn_rate: f64,
    csv_file: &str,
    debug_channel: Option<Sender<f64>>,
) {
    let (x_train, y_train, x_test, y_test) =
        get_data_csv(csv_file, 0.8)
            .expect("Could not read data from csv file");
    let mut model = Ann4::<L1, L2, L3, L3>::new(
        learn_rate,
        debug_channel,
    );

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
            train_and_validate::<2, 4, 6, 3>(
                0.8,
                "data/knn.csv",
                Some(tx),
            );
            write_costs_to_file("knn.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<2, 7, 10, 3>(
                0.99,
                "data/gda.csv",
                Some(tx),
            );
            write_costs_to_file("gda.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<8, 10, 6, 2>(
                0.001,
                "data/nb.csv",
                Some(tx),
            );
            write_costs_to_file("nb.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<2, 12, 14, 2>(
                0.8,
                "data/neg_square.csv",
                Some(tx),
            );
            write_costs_to_file("neg_square.csv", rx);
        },
        || {
            let (tx, rx) = mpsc::channel();
            train_and_validate::<2, 15, 13, 2>(
                0.5,
                "data/circle.csv",
                Some(tx),
            );
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
