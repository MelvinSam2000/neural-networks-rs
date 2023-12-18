use nalgebra::SMatrix;
use rand::Rng;

pub struct Conv2d<
    const RX: usize,
    const CX: usize,
    const RY: usize,
    const CY: usize,
    const RW: usize,
    const CW: usize,
> {
    x: SMatrix<f64, RX, CX>,
    w: SMatrix<f64, RW, CW>,
    y: SMatrix<f64, RY, CY>,
    learn_rate: f64,
}

impl<
        const RX: usize,
        const CX: usize,
        const RY: usize,
        const CY: usize,
        const RW: usize,
        const CW: usize,
    > Conv2d<RX, CX, RY, CY, RW, CW>
{
    pub fn new(learn_rate: f64) -> Self {
        assert_eq!(
            RY,
            RX - RW + 1,
            "Row dimensions for conv operation are \
             incorrect"
        );
        assert_eq!(
            CY,
            CX - CW + 1,
            "Col dimensions for conv operation are \
             incorrect"
        );

        let mut w = SMatrix::zeros();
        let x = SMatrix::zeros();
        let y = SMatrix::zeros();

        // randomize W and b
        let mut rng = rand::thread_rng();
        let uniform = rand_distr::Uniform::new(-0.5, 0.5);
        for i in 0..RX {
            for j in 0..CW {
                w[(i, j)] = rng.sample(uniform);
            }
        }

        Self {
            x,
            w,
            y,
            learn_rate,
        }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f64, RX, CX>,
    ) -> SMatrix<f64, RY, CY> {
        self.x = x;
        self.y = conv::<RX, CX, RW, CW, RY, CY>(
            &self.x, &self.w,
        );
        self.y
    }

    // backprop (TODO: find formula to propagate gradient backwards)
    pub fn bp(&mut self, g: SMatrix<f64, RY, CY>) {
        self.w -= self.learn_rate
            * conv::<RX, CX, RY, CY, RW, CW>(&self.x, &g);
    }
}

fn conv<
    const R1: usize,
    const C1: usize,
    const R2: usize,
    const C2: usize,
    const R3: usize,
    const C3: usize,
>(
    a: &SMatrix<f64, R1, C1>,
    b: &SMatrix<f64, R2, C2>,
) -> SMatrix<f64, R3, C3> {
    assert_eq!(
        R3,
        R1 - R2 + 1,
        "Row dimensions for conv operation are incorrect"
    );
    assert_eq!(
        C3,
        C1 - C2 + 1,
        "Col dimensions for conv operation are incorrect"
    );

    let mut c = SMatrix::zeros();
    for i1 in 0..R3 {
        for j1 in 0..C3 {
            for i2 in 0..R2 {
                for j2 in 0..C2 {
                    c[(i1, j1)] +=
                        a[(i1 + i2, j1 + j2)] * b[(i2, j2)];
                }
            }
        }
    }
    c
}

#[test]
#[rustfmt::skip]
fn test_conv() {

    use nalgebra::Matrix4;
    use nalgebra::Matrix3;
    use nalgebra::Matrix2;

    let a = Matrix4::new(
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
    );

    let b = Matrix2::new(
        2., 2., 
        2., 2.
    );

    let c = conv::<4, 4, 2, 2, 3, 3>(&a, &b);

    assert_eq!(c, Matrix3::new(
        8., 8., 8.,
        8., 8., 8.,
        8., 8., 8.,
    ));
}
