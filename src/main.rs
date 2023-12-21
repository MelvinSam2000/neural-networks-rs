#![feature(maybe_uninit_uninit_array, maybe_uninit_slice)]

use runners::annrun::train_and_validate_csv_ann;
use runners::cnnrun::train_and_validate_mnist_cnn;

pub mod activation;
pub mod dataset;
pub mod layers;
pub mod loss;
pub mod models;
pub mod runners;

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
