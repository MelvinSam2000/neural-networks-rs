use nalgebra::SMatrix;
use nalgebra::SVector;
use rand::Rng;

use crate::activation::deriv_all;
use crate::activation::func_all;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::tanh::Tanh;
use crate::optimizers::Optimizer;
use crate::optimizers::OptimizerFactory;

pub struct Lstm<
    const X: usize,
    const H: usize,
    const T: usize,
    const HX: usize,
    O: OptimizerFactory<H, HX> + OptimizerFactory<H, 1>,
> {
    // layer variables
    x: [SVector<f32, X>; T],
    h: [SVector<f32, H>; T],
    c: [SVector<f32, H>; T],

    // gate variables
    h_x: [SVector<f32, HX>; T],
    f: [SVector<f32, H>; T],
    i: [SVector<f32, H>; T],
    c_bar: [SVector<f32, H>; T],
    o: [SVector<f32, H>; T],
    ch: [SVector<f32, H>; T],

    // pre activation variables
    zf: [SVector<f32, H>; T],
    zi: [SVector<f32, H>; T],
    zc: [SVector<f32, H>; T],
    zo: [SVector<f32, H>; T],

    // learnable params
    wf: SMatrix<f32, H, HX>,
    wi: SMatrix<f32, H, HX>,
    wc: SMatrix<f32, H, HX>,
    wo: SMatrix<f32, H, HX>,
    bf: SVector<f32, H>,
    bi: SVector<f32, H>,
    bc: SVector<f32, H>,
    bo: SVector<f32, H>,

    // optimizers
    optwf: <O as OptimizerFactory<H, HX>>::Optimizer,
    optbf: <O as OptimizerFactory<H, 1>>::Optimizer,
    optwi: <O as OptimizerFactory<H, HX>>::Optimizer,
    optbi: <O as OptimizerFactory<H, 1>>::Optimizer,
    optwc: <O as OptimizerFactory<H, HX>>::Optimizer,
    optbc: <O as OptimizerFactory<H, 1>>::Optimizer,
    optwo: <O as OptimizerFactory<H, HX>>::Optimizer,
    optbo: <O as OptimizerFactory<H, 1>>::Optimizer,
}

impl<
        const X: usize,
        const H: usize,
        const T: usize,
        const HX: usize,
        O,
    > Lstm<X, H, T, HX, O>
where
    O: OptimizerFactory<H, HX> + OptimizerFactory<H, 1>,
{
    pub fn new() -> Self {
        let x = [SVector::zeros(); T];
        let h = [SVector::zeros(); T];
        let c = [SVector::zeros(); T];

        let h_x = [SVector::zeros(); T];
        let f = [SVector::zeros(); T];
        let i = [SVector::zeros(); T];
        let c_bar = [SVector::zeros(); T];
        let o = [SVector::zeros(); T];
        let ch = [SVector::zeros(); T];

        let zf = [SVector::zeros(); T];
        let zi = [SVector::zeros(); T];
        let zc = [SVector::zeros(); T];
        let zo = [SVector::zeros(); T];

        let mut wf = SMatrix::zeros();
        let mut wi = SMatrix::zeros();
        let mut wc = SMatrix::zeros();
        let mut wo = SMatrix::zeros();
        let mut bf = SVector::zeros();
        let mut bi = SVector::zeros();
        let mut bc = SVector::zeros();
        let mut bo = SVector::zeros();

        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-0.5, 0.5);

        for i in 0..H {
            for j in 0..HX {
                wf[(i, j)] = rng.sample(uniform);
                wi[(i, j)] = rng.sample(uniform);
                wc[(i, j)] = rng.sample(uniform);
                wo[(i, j)] = rng.sample(uniform);
            }
            bf[i] = rng.sample(uniform);
            bi[i] = rng.sample(uniform);
            bc[i] = rng.sample(uniform);
            bo[i] = rng.sample(uniform);
        }

        let optwf =
            <O as OptimizerFactory<H, HX>>::Optimizer::init(
            );
        let optbf =
            <O as OptimizerFactory<H, 1>>::Optimizer::init(
            );
        let optwi =
            <O as OptimizerFactory<H, HX>>::Optimizer::init(
            );
        let optbi =
            <O as OptimizerFactory<H, 1>>::Optimizer::init(
            );
        let optwc =
            <O as OptimizerFactory<H, HX>>::Optimizer::init(
            );
        let optbc =
            <O as OptimizerFactory<H, 1>>::Optimizer::init(
            );
        let optwo =
            <O as OptimizerFactory<H, HX>>::Optimizer::init(
            );
        let optbo =
            <O as OptimizerFactory<H, 1>>::Optimizer::init(
            );

        Self {
            x,
            h,
            c,
            h_x,
            f,
            i,
            c_bar,
            o,
            ch,
            zf,
            zi,
            zc,
            zo,
            wf,
            wi,
            wc,
            wo,
            bf,
            bi,
            bc,
            bo,
            optwf,
            optbf,
            optwi,
            optbi,
            optwc,
            optbc,
            optwo,
            optbo,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: [SVector<f32, X>; T],
    ) -> [SVector<f32, H>; T] {
        self.x = x;
        for t in 0..T {
            self.h_x[t] = if t != 0 {
                Self::concat(&self.h[t - 1], &self.x[t])
            } else {
                Self::concat(&SVector::zeros(), &self.x[t])
            };

            self.zf[t] = self.wf * self.h_x[t] + self.bf;
            self.f[t] =
                func_all::<H, 1, Sigmoid>(&self.zf[t]);

            self.zi[t] = self.wi * self.h_x[t] + self.bi;
            self.i[t] = func_all::<H, 1, Tanh>(&self.zi[t]);

            self.zc[t] = self.wc * self.h_x[t] + self.bc;
            self.c_bar[t] =
                func_all::<H, 1, Sigmoid>(&self.zc[t]);

            self.zo[t] = self.wo * self.h_x[t] + self.bo;
            self.o[t] =
                func_all::<H, 1, Sigmoid>(&self.zo[t]);

            let c1 = if t != 0 {
                self.c[t - 1].component_mul(&self.f[t])
            } else {
                SVector::zeros()
            };
            let c2 =
                self.i[t].component_mul(&self.c_bar[t]);
            self.c[t] = c1 + c2;
            self.ch[t] = func_all::<H, 1, Tanh>(&self.c[t]);

            self.h[t] =
                self.o[t].component_mul(&self.ch[t]);
        }
        self.h
    }

    // backprop
    pub fn bp(
        &mut self,
        gy: [SVector<f32, H>; T],
    ) -> [SVector<f32, X>; T] {
        let mut gx = [SVector::zeros(); T];
        let mut gc = SVector::zeros();
        let mut gh = SVector::zeros();
        let mut dwf = SMatrix::zeros();
        let mut dwi = SMatrix::zeros();
        let mut dwc = SMatrix::zeros();
        let mut dwo = SMatrix::zeros();
        let mut dbf = SVector::zeros();
        let mut dbi = SVector::zeros();
        let mut dbc = SVector::zeros();
        let mut dbo = SVector::zeros();
        for t in (0..T).rev() {
            let mut ghx = SVector::zeros();
            gh = gh + &gy[t];
            let go = gh
                .component_mul(&self.ch[t])
                .component_mul(
                    &deriv_all::<H, 1, Sigmoid>(
                        &self.zo[t],
                    ),
                );
            dwo += &go * self.h_x[t].transpose();
            dbo += &go;
            ghx += self.wo.transpose() * go;
            gc = gc + gh.component_mul(&self.o[t]);
            gc = gc.component_mul(
                &deriv_all::<H, 1, Tanh>(&self.c[t]),
            );
            let gi = gc
                .component_mul(&self.c_bar[t])
                .component_mul(&deriv_all::<H, 1, Tanh>(
                    &self.zi[t],
                ));
            dwi += &gi * self.h_x[t].transpose();
            dbi += &gi;
            ghx += self.wi.transpose() * gi;
            let gcbar =
                gc.component_mul(&self.i[t]).component_mul(
                    &deriv_all::<H, 1, Sigmoid>(
                        &self.zc[t],
                    ),
                );
            dwc += &gcbar * self.h_x[t].transpose();
            dbc += gcbar;
            ghx += self.wc.transpose() * gcbar;
            let gf = if t != 0 {
                gc.component_mul(&self.c[t - 1])
                    .component_mul(&deriv_all::<
                        H,
                        1,
                        Sigmoid,
                    >(
                        &self.zf[t]
                    ))
            } else {
                SVector::zeros()
            };
            dwf += &gf * self.h_x[t].transpose();
            dbf += &gf;
            ghx += self.wf.transpose() * gf;
            gc = gc.component_mul(&self.f[t]);
            let (tmp_gh, tmp_gx) = Self::unconcat(&ghx);
            gx[t] = tmp_gx;
            gh = tmp_gh;
        }

        self.optwf.update_param(&mut self.wf, &dwf);
        self.optbf.update_param(&mut self.bf, &dbf);
        self.optwi.update_param(&mut self.wi, &dwi);
        self.optbi.update_param(&mut self.bi, &dbi);
        self.optwc.update_param(&mut self.wc, &dwc);
        self.optbc.update_param(&mut self.bc, &dbc);
        self.optwo.update_param(&mut self.wo, &dwo);
        self.optbo.update_param(&mut self.bo, &dbo);

        gx
    }

    fn concat(
        h: &SVector<f32, H>,
        x: &SVector<f32, X>,
    ) -> SVector<f32, HX> {
        let mut out = SVector::zeros();
        for i in 0..H {
            out[i] = h[i];
        }
        for i in 0..X {
            out[H + i] = x[i];
        }
        out
    }

    fn unconcat(
        hx: &SVector<f32, HX>,
    ) -> (SVector<f32, H>, SVector<f32, X>) {
        let mut h = SVector::zeros();
        let mut x = SVector::zeros();
        for i in 0..H {
            h[i] = hx[i];
        }
        for i in 0..X {
            x[i] = hx[H + i];
        }
        (h, x)
    }
}
