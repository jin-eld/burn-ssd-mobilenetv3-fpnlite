use crate::dataset::SSDBatch;
use crate::loss::SSDLoss;
use crate::target::target_encoder::SSDTargetEncoder;
use crate::training::ssd_output::SSDOutput;

use burn::module::Module;
use burn::optim::GradientsParams;
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use burn::{tensor::Device, Tensor};

use ssd::model::SSDLiteMobileNetV3;

#[derive(Module, Debug)]
pub struct SSDTrainModel {
    pub model: SSDLiteMobileNetV3,
    #[module(skip)]
    pub anchors: Vec<[f32; 4]>,
    #[module(skip)]
    pub encoder: SSDTargetEncoder,
    #[module(skip)]
    pub loss_fn: SSDLoss,
}

pub trait SSDTraining {
    fn forward_training(
        &self,
        batch: SSDBatch,
        anchors: &Vec<[f32; 4]>,
        encoder: &SSDTargetEncoder,
        loss_fn: &SSDLoss,
    ) -> SSDOutput;
}

impl SSDTraining for SSDLiteMobileNetV3 {
    fn forward_training(
        &self,
        batch: SSDBatch,
        anchors: &Vec<[f32; 4]>,
        encoder: &SSDTargetEncoder,
        loss_fn: &SSDLoss,
    ) -> SSDOutput {
        let device = batch.images.device();

        // raw head outputs: logits + deltas
        let (pred_logits, pred_deltas) = self.forward_raw(batch.images);

        // decode deltas -> boxes using training anchors
        let anchors_tensor = anchors_to_tensor(anchors, &device);

        let pred_boxes =
            self.decoder().decode(pred_deltas.clone(), anchors_tensor);

        // encode targets
        let (tgt_classes, tgt_boxes, pos_mask) =
            encoder.encode_batch(anchors, batch.boxes, batch.labels, &device);

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

impl TrainStep for SSDTrainModel
where
    SSDLiteMobileNetV3: SSDTraining,
{
    type Input = SSDBatch;
    type Output = SSDOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let out = self.model.forward_training(
            batch,
            &self.anchors,
            &self.encoder,
            &self.loss_fn,
        );

        let raw_grads = out.loss.backward();
        let grads = GradientsParams::from_grads::<_>(raw_grads, self);

        return TrainOutput { grads, item: out };
    }
}

impl InferenceStep for SSDTrainModel
where
    SSDLiteMobileNetV3: SSDTraining,
{
    type Input = SSDBatch;
    type Output = SSDOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        return self.model.forward_training(
            batch,
            &self.anchors,
            &self.encoder,
            &self.loss_fn,
        );
    }
}

pub fn anchors_to_tensor(
    anchors: &Vec<[f32; 4]>,
    device: &Device,
) -> Tensor<2> {
    let num = anchors.len();

    // flatten into Vec<f32>
    let mut flat = Vec::with_capacity(num * 4);
    for a in anchors {
        flat.extend_from_slice(a);
    }

    // 1D tensor from &[f32]
    let t1 = Tensor::<1>::from_floats(flat.as_slice(), device);

    // reshape to [num_anchors, 4]
    return t1.reshape([num, 4]);
}
