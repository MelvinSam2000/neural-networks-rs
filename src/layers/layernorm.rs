use nalgebra::SMatrix;

#[derive(Default)]
pub struct LayerNorm<const R: usize, const C: usize> {
    sample_var: f32,
}

impl<const R: usize, const C: usize> LayerNorm<R, C> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ff(
        &mut self,
        mut x: SMatrix<f32, R, C>,
    ) -> SMatrix<f32, R, C> {
        // mean
        let mut sample_mean = 0.;
        for i in 0..R {
            for j in 0..C {
                sample_mean += x[(i, j)];
            }
        }
        sample_mean /= (R * C) as f32;
        // var
        let mut sample_var = 0.;
        for i in 0..R {
            for j in 0..C {
                sample_var +=
                    (sample_mean - x[(i, j)]).powi(2);
            }
        }
        sample_var /= (R * C) as f32 + 1.;
        *self = Self { sample_var };
        for i in 0..R {
            for j in 0..C {
                x[(i, j)] = (x[(i, j)] - sample_mean)
                    / sample_var.sqrt()
            }
        }
        x
    }

    pub fn bp(
        &self,
        x: SMatrix<f32, R, C>,
    ) -> SMatrix<f32, R, C> {
        x * (1. / self.sample_var.sqrt())
    }
}
