use std::collections::HashMap;

use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

#[derive(Default)]
// N: number of tokens (words in sentence), M: dimension of the embedding
pub struct RandEmbedding<const N: usize, const M: usize> {
    map: HashMap<String, SVector<f32, M>>,
}

impl<const N: usize, const M: usize> RandEmbedding<N, M> {
    pub fn embed(
        &mut self,
        sentence: String,
    ) -> SMatrix<f32, N, M> {
        let mut sentence = sentence.to_lowercase();
        sentence.retain(|c| {
            c.is_alphabetic() || c.is_whitespace()
        });
        let mut out: SMatrix<f32, N, M> = SMatrix::zeros();
        for (i, word) in sentence.split(' ').enumerate() {
            if i >= N {
                break;
            }
            let v = self
                .map
                .entry(word.to_string())
                .or_insert({
                    let mut v = SVector::zeros();
                    let mut rng = rand::thread_rng();
                    let uniform =
                        rand_distr::Uniform::new(-1., 1.);
                    for i in 0..M {
                        v[i] = rng.sample(uniform);
                    }
                    v
                });
            out.set_row(i, &v.transpose());
        }
        out
    }
}

#[test]
#[ignore]
fn test_rembedding() {
    let mut emb: RandEmbedding<6, 10> =
        RandEmbedding::default();
    let m = emb.embed("Hello how are you hello".into());
    println!("{m:.2}");
}
