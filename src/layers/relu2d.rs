use nalgebra::SMatrix;

pub struct Relu2dLayer<const R: usize, const C: usize> {
    m: SMatrix<f64, R, C>,
}

impl<const R: usize, const C: usize> Relu2dLayer<R, C> {
    pub fn new() -> Self {
        let m = SMatrix::zeros();
        Self { m }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f64, R, C>,
    ) -> SMatrix<f64, R, C> {
        let mut out = SMatrix::zeros();
        for i in 0..R {
            for j in 0..C {
                if x[(i, j)] > 0. {
                    out[(i, j)] = x[(i, j)];
                    self.m[(i, j)] = 1.;
                } else {
                    self.m[(i, j)] = 0.;
                }
            }
        }
        out
    }

    // backprop
    pub fn bp(
        &mut self,
        g: SMatrix<f64, R, C>,
    ) -> SMatrix<f64, R, C> {
        g.component_mul(&self.m)
    }
}
