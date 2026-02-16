use burn::tensor::{backend::Backend, Int, Tensor};
use burn::train::ItemLazy;

impl<B: Backend> ItemLazy for SSDOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        return self;
    }
}

pub struct SSDOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub loss_cls: Tensor<B, 1>,
    pub loss_reg: Tensor<B, 1>,
    pub pred_logits: Tensor<B, 3>,      // [N, A, C]
    pub pred_boxes: Tensor<B, 3>,       // [N, A, 4]
    pub tgt_classes: Tensor<B, 2, Int>, // [N, A]
    pub tgt_boxes: Tensor<B, 3>,        // [N, A, 4]
    pub pos_mask: Tensor<B, 2, Int>,    // [N, A]
}

impl<B: Backend> SSDOutput<B> {
    pub fn new(
        loss: Tensor<B, 1>,
        loss_cls: Tensor<B, 1>,
        loss_reg: Tensor<B, 1>,
        pred_logits: Tensor<B, 3>,
        pred_boxes: Tensor<B, 3>,
        tgt_classes: Tensor<B, 2, Int>,
        tgt_boxes: Tensor<B, 3>,
        pos_mask: Tensor<B, 2, Int>,
    ) -> Self {
        return Self {
            loss,
            loss_cls,
            loss_reg,
            pred_logits,
            pred_boxes,
            tgt_classes,
            tgt_boxes,
            pos_mask,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    type B = Wgpu;

    #[test]
    fn ssd_output_constructs_and_exposes_fields() {
        let device = Default::default();

        let loss = Tensor::<B, 1>::zeros([1], &device);
        let loss_cls = Tensor::<B, 1>::zeros([1], &device);
        let loss_reg = Tensor::<B, 1>::zeros([1], &device);

        let pred_logits = Tensor::<B, 3>::zeros([2, 10, 4], &device);
        let pred_boxes = Tensor::<B, 3>::zeros([2, 10, 4], &device);
        let tgt_classes = Tensor::<B, 2, Int>::zeros([2, 10], &device);
        let tgt_boxes = Tensor::<B, 3>::zeros([2, 10, 4], &device);
        let pos_mask = Tensor::<B, 2, Int>::zeros([2, 10], &device);

        let out = SSDOutput::new(
            loss,
            loss_cls,
            loss_reg,
            pred_logits,
            pred_boxes,
            tgt_classes,
            tgt_boxes,
            pos_mask,
        );

        // basic sanity: dims are as expected
        assert_eq!(out.pred_logits.dims(), [2, 10, 4]);
        assert_eq!(out.pred_boxes.dims(), [2, 10, 4]);
        assert_eq!(out.tgt_classes.dims(), [2, 10]);
        assert_eq!(out.tgt_boxes.dims(), [2, 10, 4]);
        assert_eq!(out.pos_mask.dims(), [2, 10]);
    }
}
