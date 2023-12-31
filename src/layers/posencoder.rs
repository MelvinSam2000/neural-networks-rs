use nalgebra::SMatrix;

// n tokens, m embedding dimension
pub struct PosEncoder<const N: usize, const M: usize> {
    p: SMatrix<f32, N, M>,
}

impl<const N: usize, const M: usize> PosEncoder<N, M> {
    pub fn new() -> Self {
        let mut p = SMatrix::zeros();
        const K: f32 = 10000.;
        for i in 0..N {
            for j in 0..M {
                p[(i, j)] = if i % 2 == 0 {
                    (i as f32 / K.powi((j / M) as i32))
                        .sin()
                } else {
                    (i as f32 / K.powi((j / M) as i32))
                        .cos()
                }
            }
        }
        Self { p }
    }

    pub fn ff(
        &self,
        x: SMatrix<f32, N, M>,
    ) -> SMatrix<f32, N, M> {
        x + self.p
    }
}
