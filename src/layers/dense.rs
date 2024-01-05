use nalgebra::SVector;

use super::layernorm::LayerNorm;
use super::sequential::Sequential;
use super::softmax::Softmax;
use crate::activation::noact::NoActivation;
use crate::activation::ActivationFunction;
use crate::optimizers::OptimizerFactory;

pub struct Dense<
    const X: usize,
    const Y: usize,
    const H: usize,
    const L: usize,
    F,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, 1>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>
        + OptimizerFactory<Y, 1>,
> {
    start_layer: Sequential<X, H, F, O>,
    mid_layers: Vec<Sequential<H, H, F, O>>,
    final_layer: Sequential<H, Y, NoActivation, O>,
    layernorm: LayerNorm<Y, 1>,
    softmax: Softmax<Y>,
}

impl<
        const X: usize,
        const Y: usize,
        const H: usize,
        const L: usize,
        F,
        O,
    > Dense<X, Y, H, L, F, O>
where
    F: ActivationFunction,
    O: OptimizerFactory<H, X>
        + OptimizerFactory<H, 1>
        + OptimizerFactory<H, H>
        + OptimizerFactory<Y, H>
        + OptimizerFactory<Y, 1>,
{
    pub fn new() -> Self {
        let start_layer = Sequential::new();
        let mid_layers = (0..L)
            .map(|_| Sequential::new())
            .collect::<Vec<_>>();
        let final_layer = Sequential::new();
        let layernorm = LayerNorm::new();
        let softmax = Softmax::new();
        Self {
            start_layer,
            mid_layers,
            final_layer,
            layernorm,
            softmax,
        }
    }

    pub fn ff(
        &mut self,
        x: SVector<f32, X>,
    ) -> SVector<f32, Y> {
        let mut x = self.start_layer.ff(x);
        for l in 0..L {
            x = self.mid_layers[l].ff(x);
        }
        let x = self.final_layer.ff(x);
        let x = self.layernorm.ff(x);
        let x = self.softmax.ff(x);
        x
    }

    pub fn bp(
        &mut self,
        g: SVector<f32, Y>,
    ) -> SVector<f32, X> {
        let g = self.softmax.bp(g);
        let g = self.layernorm.bp(g);
        let mut g = self.final_layer.bp(g);
        for l in (0..L).rev() {
            g = self.mid_layers[l].bp(g);
        }
        let g = self.start_layer.bp(g);
        g
    }
}
