use std::fs::File;
use std::io::BufReader;

use finalfusion::compat::word2vec::ReadWord2Vec;
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
        let mut reader = BufReader::new(
            File::open(emb_file)
                .expect("Could not open embeddings file"),
        );
        let embeddings: Embeddings<SimpleVocab, NdArray> =
            Embeddings::read_word2vec_binary(&mut reader)
                .expect("Could not parse embeddings file");
        Self { embeddings }
    }

    pub fn embed<const T: usize>(
        &self,
        words: &str,
    ) -> [SVector<f32, T>; N] {
        let words = words.split(' ').map(|word| match self
            .embeddings
            .embedding(word)
        {
            Some(x) => {
                SVector::<f32, T>::from_column_slice(
                    &x.as_slice()
                        .unwrap_or(&[0.])
                        .into_iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<_>>(),
                )
            }
            None => SVector::<f32, T>::zeros(),
        });

        let mut out = [SVector::<f32, T>::zeros(); N];
        words.take(N).enumerate().for_each(|(i, x)| {
            out[i] = x;
        });
        out
    }
}

#[test]
#[ignore]
fn test_load_embedding() {
    let embedding =
        Embedding::<1>::new("embedding/model.bin");
    println!("WORDS: {:?}", embedding.embeddings.vocab());
    /*
    println!(
        "WORD: {:?}",
        embedding.embeddings.embedding("apple_NOUN")
    ); // .embed::<300>("apple"));
       //eprintln!("{:?}", embedding.embed::<300>("orange"));
       */
}
