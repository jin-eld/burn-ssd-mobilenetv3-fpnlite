use crate::dataset::SSDBatch;
use crate::loss::SSDLoss;
use crate::target::target_encoder::SSDTargetEncoder;
use crate::training::ssd_output::SSDOutput;

use burn::module::{Ignored, Module};
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use burn::Tensor;

use ssd::model::SSDLiteMobileNetV3;

#[derive(Module, Debug)]
pub struct SSDTrainModel<B: Backend> {
    pub model: SSDLiteMobileNetV3<B>,
    pub anchors: Ignored<Vec<[f32; 4]>>,
    pub encoder: Ignored<SSDTargetEncoder>,
    pub loss_fn: Ignored<SSDLoss>,
    pub device: Ignored<B::Device>,
}

pub trait SSDTraining<B: Backend> {
    fn forward_training(
        &self,
        batch: SSDBatch<B>,
        anchors: &Vec<[f32; 4]>,
        encoder: &SSDTargetEncoder,
        loss_fn: &SSDLoss,
        device: &B::Device,
    ) -> SSDOutput<B>;
}

impl<B: Backend> SSDTraining<B> for SSDLiteMobileNetV3<B> {
    fn forward_training(
        &self,
        batch: SSDBatch<B>,
        anchors: &Vec<[f32; 4]>,
        encoder: &SSDTargetEncoder,
        loss_fn: &SSDLoss,
        device: &B::Device,
    ) -> SSDOutput<B> {
        // raw head outputs: logits + deltas
        let (pred_logits, pred_deltas) = self.forward_raw(batch.images);

        // decode deltas -> boxes using training anchors
        let anchors_tensor = anchors_to_tensor::<B>(anchors, device);

        let pred_boxes = self
            .decoder()
            .decode::<B>(pred_deltas.clone(), anchors_tensor);

        let pred_boxes =
            Tensor::<B, 3>::from_data(pred_boxes.to_data(), device);

        // encode targets
        let (tgt_classes, tgt_boxes, pos_mask) = encoder.encode_batch::<B>(
            anchors,
            batch.boxes,
            batch.labels,
            device,
        );

        // compute loss on decoded boxes
        let (loss_total, loss_cls, loss_reg) = loss_fn.forward(
            pred_logits.clone(),
            pred_boxes.clone(),
            tgt_classes.clone(),
            tgt_boxes.clone(),
            pos_mask.clone(),
        );

        return SSDOutput::new(
            loss_total,
            loss_cls,
            loss_reg,
            pred_logits,
            pred_boxes,
            tgt_classes,
            tgt_boxes,
            pos_mask,
        );
    }
}

impl<B: AutodiffBackend> TrainStep for SSDTrainModel<B>
where
    SSDLiteMobileNetV3<B>: SSDTraining<B>,
{
    type Input = SSDBatch<B>;
    type Output = SSDOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let out = self.model.forward_training(
            batch,
            &self.anchors.0,
            &self.encoder.0,
            &self.loss_fn.0,
            &self.device.0,
        );

        let raw_grads = out.loss.backward();
        let grads = GradientsParams::from_grads::<B, _>(raw_grads, self);

        return TrainOutput { grads, item: out };
    }
}

impl<B: Backend> InferenceStep for SSDTrainModel<B>
where
    SSDLiteMobileNetV3<B>: SSDTraining<B>,
{
    type Input = SSDBatch<B>;
    type Output = SSDOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        return self.model.forward_training(
            batch,
            &self.anchors.0,
            &self.encoder.0,
            &self.loss_fn.0,
            &self.device.0,
        );
    }
}

pub fn anchors_to_tensor<B: Backend>(
    anchors: &Vec<[f32; 4]>,
    device: &B::Device,
) -> Tensor<B, 2> {
    let num = anchors.len();

    // flatten into Vec<f32>
    let mut flat = Vec::with_capacity(num * 4);
    for a in anchors {
        flat.extend_from_slice(a);
    }

    // 1D tensor from &[f32]
    let t1 = Tensor::<B, 1>::from_floats(flat.as_slice(), device);

    // reshape to [num_anchors, 4]
    return t1.reshape([num, 4]);
}
