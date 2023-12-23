use nalgebra::SMatrix;

#[derive(Clone, Copy)]
pub struct MaxPool2d<
    const RX: usize,
    const CX: usize,
    const RY: usize,
    const CY: usize,
    const RW: usize,
    const CW: usize,
> {
    m: [[(usize, usize); CY]; RY],
}

impl<
        const RX: usize,
        const CX: usize,
        const RY: usize,
        const CY: usize,
        const RW: usize,
        const CW: usize,
    > MaxPool2d<RX, CX, RY, CY, RW, CW>
{
    pub fn new() -> Self {
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
        let m = [[(0, 0); CY]; RY];
        Self { m }
    }

    // feedforward
    pub fn ff(
        &mut self,
        x: SMatrix<f64, RX, CX>,
    ) -> SMatrix<f64, RY, CY> {
        maxpool(&x, &mut self.m)
    }

    // backprop
    pub fn bp(
        &mut self,
        g: SMatrix<f64, RY, CY>,
    ) -> SMatrix<f64, RX, CX> {
        let mut out = SMatrix::zeros();
        for i in 0..RY {
            for j in 0..CY {
                out[self.m[i][j]] += g[(i, j)];
            }
        }
        out
    }
}

fn maxpool<
    const RX: usize,
    const CX: usize,
    const RY: usize,
    const CY: usize,
>(
    a: &SMatrix<f64, RX, CX>,
    m: &mut [[(usize, usize); CY]; RY],
) -> SMatrix<f64, RY, CY> {
    let mut c = SMatrix::zeros();
    for i1 in 0..RY {
        for j1 in 0..CY {
            let mut curmax = f64::MIN;
            let mut curidx = (0, 0);
            for i2 in 0..RX - RY + 1 {
                for j2 in 0..CX - CY + 1 {
                    if a[(i1 + i2, j1 + j2)] > curmax {
                        curmax = a[(i1 + i2, j1 + j2)];
                        curidx = (i1 + i2, j1 + j2);
                    }
                }
            }
            m[i1][j1] = curidx;
            c[(i1, j1)] = curmax;
        }
    }
    c
}

#[test]
#[rustfmt::skip]
fn test_conv() {

    use nalgebra::Matrix4;
    use nalgebra::Matrix3;

    let x = Matrix4::new(
        8., 2., 1., 9.,
        1., 1., 3., 1.,
        1., 2., 1., 1.,
        10., 1., 1., 11.,
    );

    let mut mi = [
        [(0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0)],
        [(0, 0), (0, 0), (0, 0)],
    ];
    let mf = [
        [(0, 0), (1, 2), (0, 3)],
        [(2, 1), (1, 2), (1, 2)],
        [(3, 0), (2, 1), (3, 3)],
    ];
    let y = maxpool(&x, &mut mi);

    let yf  = Matrix3::new(
        8., 3., 9.,
        2., 3., 3.,
        10., 2., 11.,
    );

    assert_eq!(mi, mf);
    assert_eq!(y, yf);


}
