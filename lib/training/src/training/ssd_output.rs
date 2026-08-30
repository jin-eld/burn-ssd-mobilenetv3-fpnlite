use burn::tensor::{Float, Int, Tensor};
use burn::train::{
    metric::{
        state::{FormatOptions, NumericMetricState},
        Adaptor, LossInput, Metric, MetricMetadata, Numeric, NumericEntry,
        SerializedEntry,
    },
    ItemLazy,
};
use std::sync::Arc;

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

// diagnostic metrics
#[derive(Clone)]
pub struct ClsLossInput(pub Tensor<1, Float>);
impl ItemLazy for ClsLossInput {
    fn sync(self) -> Self {
        return self;
    }
}

#[derive(Clone)]
pub struct RegLossInput(pub Tensor<1, Float>);
impl ItemLazy for RegLossInput {
    fn sync(self) -> Self {
        return self;
    }
}

#[derive(Clone)]
pub struct PosCountInput(pub Tensor<1, Float>);
impl ItemLazy for PosCountInput {
    fn sync(self) -> Self {
        return self;
    }
}

impl Adaptor<ClsLossInput> for SSDOutput {
    fn adapt(&self) -> ClsLossInput {
        return ClsLossInput(self.loss_cls.clone());
    }
}

impl Adaptor<RegLossInput> for SSDOutput {
    fn adapt(&self) -> RegLossInput {
        return RegLossInput(self.loss_reg.clone());
    }
}

impl Adaptor<PosCountInput> for SSDOutput {
    fn adapt(&self) -> PosCountInput {
        // Sum all positive masks in the batch to get total positive anchors
        let count = self.pos_mask.clone().float().flatten::<1>(0, 1).sum_dim(0);
        return PosCountInput(count);
    }
}

pub trait ScalarExtractor {
    fn get_scalar(&self) -> f64;
}

impl ScalarExtractor for ClsLossInput {
    fn get_scalar(&self) -> f64 {
        return self.0.clone().into_scalar::<f32>() as f64;
    }
}

impl ScalarExtractor for RegLossInput {
    fn get_scalar(&self) -> f64 {
        return self.0.clone().into_scalar::<f32>() as f64;
    }
}

impl ScalarExtractor for PosCountInput {
    fn get_scalar(&self) -> f64 {
        return self.0.clone().into_scalar::<f32>() as f64;
    }
}

#[derive(Clone)]
pub struct AvgScalarMetric<I: ScalarExtractor + Send + Sync + Clone + 'static> {
    state: NumericMetricState,
    name: Arc<String>,
    _marker: core::marker::PhantomData<I>,
}

impl<I: ScalarExtractor + Send + Sync + Clone + 'static> AvgScalarMetric<I> {
    pub fn new(name: &str) -> Self {
        return Self {
            state: NumericMetricState::new(),
            name: Arc::new(name.to_string()),
            _marker: core::marker::PhantomData,
        };
    }
}

impl<I: ScalarExtractor + Send + Sync + Clone + 'static> Metric
    for AvgScalarMetric<I>
{
    type Input = I;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
        let val = item.get_scalar();
        self.state.update(val, 1);
        return self
            .state
            .compute_update(FormatOptions::new(self.name.clone()));
    }

    fn compute(&mut self) -> SerializedEntry {
        return self
            .state
            .compute_final(FormatOptions::new(self.name.clone()));
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> Arc<String> {
        return self.name.clone();
    }
}

impl<I: ScalarExtractor + Send + Sync + Clone + 'static> Numeric
    for AvgScalarMetric<I>
{
    fn value(&self) -> Option<NumericEntry> {
        return Some(self.state.current_value());
    }

    fn running_value(&self) -> Option<NumericEntry> {
        return Some(self.state.running_value());
    }

    fn final_value(&self) -> NumericEntry {
        return self.state.final_value();
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
