use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::sync::mpsc;

use nalgebra::SMatrix;
use nalgebra::SVector;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPoolBuilder;
use regex::Regex;

use crate::layers::randembedding::RandEmbedding;
use crate::models::transformer1::Transformer1;
use crate::models::NNClassifierModel;
use crate::optimizers::adam::AdamFactory;
use crate::runners::write_costs_to_file;

const N: usize = 100;
const M: usize = 5;

pub fn train_and_validate_imdb_rnn() {
    let pool = ThreadPoolBuilder::new()
        .stack_size(1024 * 1024 * 1024)
        .build()
        .unwrap();

    println!("Data preprocessing START");

    let (x_train, y_train, x_test, y_test) =
        get_data_csv("data/imdb.csv", 0.8)
            .expect("Could not read data from csv file");

    let y_test = y_test
        .into_iter()
        .map(|v| if v[0] != 0. { 0 } else { 1 })
        .collect::<Vec<usize>>();

    println!("Data preprocessing DONE");

    pool.install(|| {
        (0..6)
            .into_par_iter()
            .map(|i| {
                let (tx, rx) = mpsc::channel();
                let mut model =
                    NNClassifierModel::<
                        /*
                        RnnSentimentAnalyzer<
                            300,
                            300,
                            2,
                            AdamFactory<
                                1,
                                100,
                                95,
                                100,
                                95,
                                100,
                            >,
                        >,
                        */
                        Transformer1<
                            AdamFactory<
                                1,
                                100,
                                95,
                                100,
                                95,
                                100,
                            >,
                        >,
                        2,
                    >::new(Some(tx));
                let dbg_thread =
                    std::thread::spawn(move || {
                        write_costs_to_file(
                            &format!("rnn-{i}.txt"),
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
                    "RNN Score thread-{id}: {:.3}%\t",
                    score * 100.
                );
                dbg_thread.join().unwrap();
            });
    });
}

fn get_data_csv(
    file_path: &str,
    train_test_ratio: f32,
) -> anyhow::Result<(
    Vec<SMatrix<f32, N, M>>,
    Vec<SVector<f32, 2>>,
    Vec<SMatrix<f32, N, M>>,
    Vec<SVector<f32, 2>>,
)> {
    let file = File::open(file_path)?;
    let filereader = BufReader::new(file);
    let fileiter = filereader
        .lines()
        .skip(1)
        .map(|line| line.unwrap())
        .collect::<Vec<String>>();
    let train_limit =
        (fileiter.len() as f32 * train_test_ratio) as usize;

    let mut x_train = Vec::new();
    let mut y_train = Vec::<SVector<f32, 2>>::new();
    let mut x_test = Vec::new();
    let mut y_test = Vec::<SVector<f32, 2>>::new();

    let mut embed = RandEmbedding::default();

    let re = Regex::new(r#"^"?(.*)"?,(\w+)"#).unwrap();

    fileiter.into_iter().enumerate().for_each(
        |(i, line)| {
            let (x, y) = if i < train_limit {
                (&mut x_train, &mut y_train)
            } else {
                (&mut x_test, &mut y_test)
            };
            let text = re
                .captures(&line)
                .expect(&format!("Failed parsing at {i}"))
                .get(1)
                .expect(&format!("Failed parsing at {i}"))
                .as_str()
                .to_string();
            let sentiment = re
                .captures(&line)
                .expect(&format!("Failed parsing at {i}"))
                .get(2)
                .expect(&format!("Failed parsing at {i}"))
                .as_str();
            let sentiment = match sentiment {
                "positive" => {
                    SVector::<f32, 2>::new(1., 0.)
                }
                "negative" => {
                    SVector::<f32, 2>::new(0., 1.)
                }
                _ => panic!(
                    "Error parsing sentiment: {}",
                    sentiment
                ),
            };
            let v = embed.embed(text);
            x.push(v);
            y.push(sentiment);
        },
    );

    Ok((x_train, y_train, x_test, y_test))
}
