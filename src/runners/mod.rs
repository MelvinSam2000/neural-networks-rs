use std::fs::File;
use std::io::Write;
use std::sync::mpsc::Receiver;

use csv::ReaderBuilder;

pub mod annrun;
pub mod cnnrun;
pub mod rnnrun;

fn write_costs_to_file(file: &str, recv: Receiver<f32>) {
    let mut costs = vec![];
    while let Ok(cost) = recv.recv() {
        costs.push(cost);
    }
    let mut destfile =
        File::create(format!("debug/{file}"))
            .expect("Please run 'mkdir debug'");
    for cost in costs {
        writeln!(destfile, "{cost}").unwrap();
    }
}

pub fn get_data_csv<const X: usize>(
    file_path: &str,
    train_test_ratio: f32,
) -> anyhow::Result<(
    Vec<[f32; X]>,
    Vec<usize>,
    Vec<[f32; X]>,
    Vec<usize>,
)> {
    let file = File::open(file_path)?;
    let records = ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(file)
        .records()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    let train_limit =
        (records.len() as f32 * train_test_ratio) as usize;

    let mut x_train = Vec::<[f32; X]>::new();
    let mut y_train = Vec::<usize>::new();
    let mut x_test = Vec::<[f32; X]>::new();
    let mut y_test = Vec::<usize>::new();

    let mut i = 0;
    while i < train_limit {
        let mut x = [0.0; X];
        for j in 0..X {
            x[j] = records[i][j].parse::<f32>()?;
        }
        let y = records[i][X].parse::<usize>()?;
        x_train.push(x);
        y_train.push(y);
        i += 1;
    }
    while i < records.len() {
        let mut x = [0.0; X];
        for j in 0..X {
            x[j] = records[i][j].parse::<f32>()?;
        }
        let y = records[i][X].parse::<usize>()?;
        x_test.push(x);
        y_test.push(y);
        i += 1;
    }

    Ok((x_train, y_train, x_test, y_test))
}
