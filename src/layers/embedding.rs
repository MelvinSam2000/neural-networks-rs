use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use finalfusion::compat::word2vec::ReadWord2Vec;
use finalfusion::embeddings::Embeddings;
use finalfusion::storage::NdArray;
use finalfusion::vocab::SimpleVocab;
use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;
use rand_distr::num_traits::Zero;

pub struct Embedding<const N: usize, const M: usize> {
    embeddings: Embeddings<SimpleVocab, NdArray>,
    randomemb: HashMap<String, SVector<f32, M>>,
}

impl<const N: usize, const M: usize> Embedding<N, M> {
    pub fn new(emb_file: &str) -> Self {
        let mut reader = BufReader::new(
            File::open(emb_file)
                .expect("Could not open embeddings file"),
        );
        let embeddings: Embeddings<SimpleVocab, NdArray> =
            Embeddings::read_word2vec_binary(&mut reader)
                .expect("Could not parse embeddings file");
        let randomemb = HashMap::default();
        Self {
            embeddings,
            randomemb,
        }
    }

    pub fn embed(
        &mut self,
        sentence: String,
    ) -> SMatrix<f32, N, M> {
        let part_of_speech =
            ["NOUN", "PROPN", "ADJ", "NUM"];

        let mut sentence = sentence.to_lowercase();
        sentence.retain(|c| {
            c.is_alphabetic() || c.is_whitespace()
        });
        let mut out: SMatrix<f32, N, M> = SMatrix::zeros();
        sentence
            .split(' ')
            .map(|token| {

                if let Some(v) = self.randomemb.get(token) {
                    return v.clone();
                }

                for pos in part_of_speech {
                    let token_pos = format!("{token}_{pos}");
                    if let Some(v) = self.embeddings.embedding(&token_pos) {
                        return SVector::<f32, M>::from_column_slice(
                            &v.as_slice()
                                .unwrap_or(&[0.])
                                .into_iter()
                                .map(|&x| x as f32)
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                let mut rng = rand::thread_rng();
                let uniform = rand_distr::Uniform::new(-0.5, 0.5);
                let mut v = SVector::zero();
                for i in 0..M {
                    v[i] = rng.sample(uniform);
                }
                self.randomemb.insert(token.to_string(), v);
                v
            })
            .take(N)
            .enumerate()
            .for_each(|(i, x)| {
                out.set_row(i, &x.transpose());
            });
        out
    }
}

#[test]
#[ignore]
fn test_load_embedding() {
    let mut embedding =
        Embedding::<7, 300>::new("embedding/model.bin");

    println!(
        "{}",
        embedding.embed("Hi I ate an apple".to_string())
    );

    println!("NOT FOUND: {:?}", embedding.randomemb.keys());
}
