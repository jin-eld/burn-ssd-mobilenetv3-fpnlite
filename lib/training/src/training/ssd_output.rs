use burn::tensor::{Float, Int, Tensor};
use burn::train::{
    metric::{Adaptor, LossInput},
    ItemLazy,
};

pub struct SSDOutput {
    pub loss: Tensor<1, Float>,
    pub loss_cls: Tensor<1, Float>,
    pub loss_reg: Tensor<1, Float>,
    pub pred_logits: Tensor<3, Float>, // [N, A, C]
    pub pred_boxes: Tensor<3, Float>,  // [N, A, 4]
    pub tgt_classes: Tensor<2, Int>,   // [N, A]
    pub tgt_boxes: Tensor<3, Float>,   // [N, A, 4]
    pub pos_mask: Tensor<2, Int>,      // [N, A]
}

impl ItemLazy for SSDOutput {
    fn sync(self) -> Self {
        return self;
    }
}

impl Adaptor<LossInput> for SSDOutput {
    fn adapt(&self) -> LossInput {
        return LossInput::new(self.loss.clone());
    }
}

impl SSDOutput {
    pub fn new(
        loss: Tensor<1, Float>,
        loss_cls: Tensor<1, Float>,
        loss_reg: Tensor<1, Float>,
        pred_logits: Tensor<3, Float>,
        pred_boxes: Tensor<3, Float>,
        tgt_classes: Tensor<2, Int>,
        tgt_boxes: Tensor<3, Float>,
        pos_mask: Tensor<2, Int>,
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

    #[test]
    fn ssd_output_constructs_and_exposes_fields() {
        let device = Default::default();

        let loss = Tensor::<1>::zeros([1], &device);
        let loss_cls = Tensor::<1>::zeros([1], &device);
        let loss_reg = Tensor::<1>::zeros([1], &device);

        let pred_logits = Tensor::<3>::zeros([2, 10, 4], &device);
        let pred_boxes = Tensor::<3>::zeros([2, 10, 4], &device);
        let tgt_classes = Tensor::<2, Int>::zeros([2, 10], &device);
        let tgt_boxes = Tensor::<3>::zeros([2, 10, 4], &device);
        let pos_mask = Tensor::<2, Int>::zeros([2, 10], &device);

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
