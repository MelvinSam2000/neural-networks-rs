use crate::ann::Ann4;
use crate::dataset::get_data_csv;

pub mod activation;
pub mod ann;
pub mod dataset;
pub mod sequential;

fn train_and_validate<
    const L1: usize,
    const L2: usize,
    const L3: usize,
    const L4: usize,
>(
    learn_rate: f64,
    csv_file: &str,
) {
    let (x_train, y_train, x_test, y_test) =
        get_data_csv(csv_file, 0.8)
            .expect("Could not read data from csv file");
    let mut model = Ann4::<L1, L2, L3, L3>::new(learn_rate);

    model.train(&x_train, &y_train);
    let score = model.validate(&x_test, &y_test);

    println!(
        "File: {}\t LR:{}\t Score: {:.3}%\t",
        csv_file,
        learn_rate,
        score * 100.
    );
}

fn main() {
    let tasks = vec![
        || {
            train_and_validate::<2, 4, 6, 3>(
                0.8,
                "data/knn.csv",
            )
        },
        || {
            train_and_validate::<2, 7, 10, 3>(
                0.1,
                "data/gda.csv",
            )
        },
        || {
            train_and_validate::<8, 4, 6, 2>(
                0.9,
                "data/nb.csv",
            )
        },
    ];
    tasks
        .into_iter()
        .skip(1)
        .take(1)
        .map(|task| std::thread::spawn(task))
        .for_each(|task| {
            task.join().expect("A task failed")
        });
}
