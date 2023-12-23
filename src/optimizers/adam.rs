use nalgebra::SMatrix;

use super::component_invsqrt;
use super::Optimizer;
use super::OptimizerFactory;

pub struct Adam<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const BETA1_NUM: usize,
    const BETA1_DEN: usize,
    const BETA2_NUM: usize,
    const BETA2_DEN: usize,
    const R: usize,
    const C: usize,
> {
    m: SMatrix<f64, R, C>,
    v: SMatrix<f64, R, C>,
    t: i32,
}

impl<
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const BETA1_NUM: usize,
        const BETA1_DEN: usize,
        const BETA2_NUM: usize,
        const BETA2_DEN: usize,
        const R: usize,
        const C: usize,
    > Optimizer<R, C>
    for Adam<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA1_NUM,
        BETA1_DEN,
        BETA2_NUM,
        BETA2_DEN,
        R,
        C,
    >
{
    fn init() -> Self {
        let m = SMatrix::zeros();
        let v = SMatrix::zeros();
        let t = 1;
        Self { m, v, t }
    }

    fn update_param(
        &mut self,
        weight: &mut SMatrix<f64, R, C>,
        gradient: &SMatrix<f64, R, C>,
    ) {
        const T: i32 = 50;
        let alpha = ALPHA_NUM as f64 / ALPHA_DEN as f64;
        let beta1 = BETA1_NUM as f64 / BETA1_DEN as f64;
        let beta2 = BETA2_NUM as f64 / BETA2_DEN as f64;

        self.m = beta1 * self.m + (1. - beta1) * gradient;
        self.v = beta2 * self.v
            + (1. - beta2)
                * gradient.component_mul(gradient);

        //let m = self.m;
        //let v = self.v;
        //println!("b1 {}", (1. / (1. - beta1.powi(self.t))));
        //println!("b2 {}", (1. / (1. - beta2.powi(self.t))));

        let m = if self.t <= T {
            self.m / (1. - beta1.powi(self.t))
        } else {
            self.m
        };
        let v = if self.t <= T {
            self.v / (1. - beta2.powi(self.t))
        } else {
            self.v
        };

        *weight -=
            alpha * component_invsqrt(&v).component_mul(&m);

        if self.t <= T {
            self.t += 1;
        }
    }

    fn name() -> String {
        "adam".to_string()
    }
}

pub struct AdamFactory<
    const ALPHA_NUM: usize,
    const ALPHA_DEN: usize,
    const BETA1_NUM: usize,
    const BETA1_DEN: usize,
    const BETA2_NUM: usize,
    const BETA2_DEN: usize,
>;

impl<
        const R: usize,
        const C: usize,
        const ALPHA_NUM: usize,
        const ALPHA_DEN: usize,
        const BETA1_NUM: usize,
        const BETA1_DEN: usize,
        const BETA2_NUM: usize,
        const BETA2_DEN: usize,
    > OptimizerFactory<R, C>
    for AdamFactory<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA1_NUM,
        BETA1_DEN,
        BETA2_NUM,
        BETA2_DEN,
    >
{
    type Optimizer = Adam<
        ALPHA_NUM,
        ALPHA_DEN,
        BETA1_NUM,
        BETA1_DEN,
        BETA2_NUM,
        BETA2_DEN,
        R,
        C,
    >;
}
