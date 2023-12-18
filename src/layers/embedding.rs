use std::fs::File;

use finalfusion::embeddings::Embeddings;
use finalfusion::io::ReadEmbeddings;
use finalfusion::storage::NdArray;
use finalfusion::vocab::SimpleVocab;
use nalgebra::SVector;

pub struct Embedding<const N: usize> {
    embeddings: Embeddings<SimpleVocab, NdArray>,
}

impl<const N: usize> Embedding<N> {
    pub fn new(emb_file: &str) -> Self {
        let embeddings: Embeddings<SimpleVocab, NdArray> =
            Embeddings::read_embeddings(
                &mut File::open(emb_file).expect(
                    "Could not open embeddings file",
                ),
            )
            .expect("Could not parse embeddings file");
        Self { embeddings }
    }

    pub fn embed<const T: usize>(
        &self,
        words: &str,
    ) -> [SVector<f64, T>; N] {
        let words = words.split(' ').map(|word| match self
            .embeddings
            .embedding(word)
        {
            Some(x) => {
                SVector::<f64, T>::from_column_slice(
                    &x.as_slice()
                        .unwrap_or(&[0.])
                        .into_iter()
                        .map(|&x| x as f64)
                        .collect::<Vec<_>>(),
                )
            }
            None => SVector::<f64, T>::zeros(),
        });

        let mut out = [SVector::<f64, T>::zeros(); N];
        words.take(N).enumerate().for_each(|(i, x)| {
            out[i] = x;
        });
        out
    }
}

#[test]
#[ignore]
fn test_load_embedding() {
    let embedding = Embedding::<10>::new(
        "embedding/google_word2vec.bin",
    );
    eprintln!(
        "{:?}",
        embedding.embed::<4>("Hello how are you?")
    );
    panic!("Panicked on purpose");
}
