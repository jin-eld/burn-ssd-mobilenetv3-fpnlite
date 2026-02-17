use burn::{
    nn::loss::HuberLossConfig,
    tensor::activation::sigmoid,
    tensor::{Int, Tensor},
};

#[derive(Clone, Debug)]
pub struct SSDLoss {
    pub cls_weight: f32,
    pub reg_weight: f32,
}

impl SSDLoss {
    pub fn new(cls_weight: f32, reg_weight: f32) -> Self {
        return Self {
            cls_weight,
            reg_weight,
        };
    }

    pub fn forward(
        &self,
        pred_logits: Tensor<3>,      // [N, A, C]
        pred_boxes: Tensor<3>,       // [N, A, 4]
        tgt_classes: Tensor<2, Int>, // [N, A]
        tgt_boxes: Tensor<3>,        // [N, A, 4]
        pos_mask: Tensor<2, Int>,    // [N, A]
    ) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
        let cls_loss = sigmoid_focal_loss(pred_logits, tgt_classes, 0.25, 2.0);
        let reg_loss = ssd_regression_loss(pred_boxes, tgt_boxes, pos_mask);

        let total = cls_loss.clone() * self.cls_weight
            + reg_loss.clone() * self.reg_weight;

        return (total, cls_loss, reg_loss);
    }
}

/// Sigmoid focal loss (RetinaNet/SSD style).
///
/// logits:  [N, A, C]  Float
/// targets: [N, A]     Int class indices
///
/// Returns: [1] Float (scalar loss)
pub fn sigmoid_focal_loss(
    logits: Tensor<3>,       // [N, A, C]
    targets: Tensor<2, Int>, // [N, A]
    alpha: f32,
    gamma: f32,
) -> Tensor<1> {
    let device = logits.device();
    let num_classes = logits.dims()[2];
    let eps = 1e-6;

    // one-hot encode targets: [N, A, C] Float
    let targets_oh: Tensor<3> =
        targets
            .clone()
            .float()
            .one_hot_fill(num_classes, 1.0, 0.0, -1);

    // sigmoid probabilities
    let p_raw: Tensor<3> = sigmoid(logits.clone());

    // clamp probabilities for numerical stability
    let p: Tensor<3> = p_raw.clamp(eps, 1.0 - eps);

    // p_t = p if target=1 else (1-p)
    let p_t_raw: Tensor<3> = p.clone() * targets_oh.clone()
        + (1.0 - p.clone()) * (1.0 - targets_oh.clone());

    let p_t: Tensor<3> = p_t_raw.clamp(eps, 1.0 - eps);

    // alpha_t = alpha for positives, (1-alpha) for negatives
    let alpha_t: Tensor<3> =
        targets_oh.clone() * alpha + (1.0 - targets_oh.clone()) * (1.0 - alpha);

    // build a tensor of ones with the same shape as targets_oh
    let ones: Tensor<3> = Tensor::ones(targets_oh.dims(), &device);

    // (1 - p)
    let one_minus_p: Tensor<3> =
        (ones.clone() - p.clone()).clamp(eps, 1.0 - eps);

    // (1 - p_t)
    let one_minus_pt: Tensor<3> = (ones - p_t.clone()).clamp(eps, 1.0 - eps);

    // (1 - p_t)^gamma
    let focal_factor: Tensor<3> = one_minus_pt.powf_scalar(gamma);

    // focal weight = alpha_t * (1 - p_t)^gamma
    let focal_weight: Tensor<3> = alpha_t * focal_factor;

    // Manual BCE (per element):
    //
    // CE = -[ y*log(p) + (1-y)*log(1-p) ]
    //
    let one_minus_targets: Tensor<3> =
        (Tensor::ones(targets_oh.dims(), &device) - targets_oh.clone())
            .clamp(eps, 1.0 - eps);

    let ce: Tensor<3> =
        -(targets_oh.clone() * p.log() + one_minus_targets * one_minus_p.log());

    // final focal loss
    return (focal_weight * ce).mean();
}

/// SSD-style masked Huber (Smooth L1) regression loss.
///
/// pred:   [N, A, 4]  predicted box deltas
/// target: [N, A, 4]  target box deltas
/// mask:   [N, A]     1 for positive anchors, 0 otherwise
///
/// Returns: [1] Float (scalar loss)
pub fn ssd_regression_loss(
    pred: Tensor<3>,
    target: Tensor<3>,
    mask: Tensor<2, Int>,
) -> Tensor<1> {
    let eps = 1e-6;

    // huber with delta = 1.0 -> classic Smooth L1
    let huber = HuberLossConfig::new(1.0).init();

    // elementwise Huber loss: [N, A, 4]
    let per_elem = huber.forward_no_reduction(pred, target);

    // broadcast mask to [N, A, 4]
    let mask_f: Tensor<3> = mask.clone().float().unsqueeze_dim(2);

    // apply mask
    let masked = per_elem * mask_f;

    // sum all regression losses → [1]
    let loss_sum = masked.sum();

    // number of positive anchors → [1]
    let num_pos = mask.float().sum();

    // normalize, avoid division by zero
    return loss_sum / (num_pos + eps);
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::{Int, Tensor};

    fn scalar_value(t: Tensor<1>) -> f32 {
        let data = t.to_data();
        let vec = data.to_vec::<f32>().unwrap();
        vec[0]
    }

    #[test]
    fn focal_loss_low_when_predictions_are_correct() {
        let device = Default::default();

        // logits strongly favor class 1
        let logits = Tensor::<3>::from_floats([[[-5.0, 5.0]]], &device);

        // target is class 1
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = sigmoid_focal_loss(logits, targets, 0.25, 2.0);
        let value = scalar_value(loss);

        assert!(value < 0.01, "loss should be small, got {}", value);
    }

    #[test]
    fn focal_loss_high_when_predictions_are_wrong() {
        let device = Default::default();

        // logits strongly favor class 0
        let logits = Tensor::<3>::from_floats([[[5.0, -5.0]]], &device);

        // target is class 1
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = sigmoid_focal_loss(logits, targets, 0.25, 2.0);
        let value = scalar_value(loss);

        assert!(value > 1.0, "loss should be large, got {}", value);
    }

    #[test]
    fn focal_loss_returns_scalar() {
        let device = Default::default();

        let logits = Tensor::<3>::zeros([2, 3, 4], &device);
        let targets = Tensor::<2, Int>::zeros([2, 3], &device);

        let loss = sigmoid_focal_loss(logits, targets, 0.25, 2.0);

        assert_eq!(loss.dims(), [1], "loss must be a scalar");
    }

    #[test]
    fn focal_loss_multiclass_consistency() {
        let device = Default::default();

        // 3 classes
        let logits = Tensor::<3>::from_floats([[[0.1, 0.2, 0.3]]], &device);

        let targets = Tensor::<2, Int>::from_ints([[2]], &device);

        let loss = sigmoid_focal_loss(logits, targets, 0.25, 2.0);
        let value = scalar_value(loss);

        assert!(value.is_finite(), "loss must be finite");
    }

    #[test]
    fn focal_loss_decreases_when_logits_move_toward_target() {
        let device = Default::default();

        // target is class 1
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        // bad prediction
        let logits_bad = Tensor::<3>::from_floats([[[5.0, -5.0]]], &device);
        let loss_bad = scalar_value(sigmoid_focal_loss(
            logits_bad,
            targets.clone(),
            0.25,
            2.0,
        ));

        // better prediction
        let logits_good = Tensor::<3>::from_floats([[[-5.0, 5.0]]], &device);
        let loss_good =
            scalar_value(sigmoid_focal_loss(logits_good, targets, 0.25, 2.0));

        assert!(
            loss_good < loss_bad,
            "loss should decrease as logits improve"
        );
    }

    #[test]
    fn focal_loss_is_symmetric_for_class_flipping() {
        let device = Default::default();

        // logits favor class 0
        let logits_a = Tensor::<3>::from_floats([[[5.0, -5.0]]], &device);
        let targets_a = Tensor::<2, Int>::from_ints([[0]], &device);

        // logits favor class 1 (mirror)
        let logits_b = Tensor::<3>::from_floats([[[-5.0, 5.0]]], &device);
        let targets_b = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss_a =
            scalar_value(sigmoid_focal_loss(logits_a, targets_a, 0.25, 2.0));
        let loss_b =
            scalar_value(sigmoid_focal_loss(logits_b, targets_b, 0.25, 2.0));

        assert!((loss_a - loss_b).abs() < 1e-6, "loss should be symmetric");
    }

    #[test]
    fn focal_loss_handles_extreme_logits() {
        let device = Default::default();

        let logits = Tensor::<3>::from_floats([[[100.0, -100.0]]], &device);
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = scalar_value(sigmoid_focal_loss(logits, targets, 0.25, 2.0));

        assert!(
            loss.is_finite(),
            "loss must remain finite for extreme logits"
        );
    }

    #[test]
    fn focal_loss_zero_when_gamma_zero_and_perfect_prediction() {
        let device = Default::default();

        let logits = Tensor::<3>::from_floats([[[-10.0, 10.0]]], &device);
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = scalar_value(sigmoid_focal_loss(logits, targets, 0.25, 0.0));

        assert!(
            loss < 1e-4,
            "loss should be near zero when gamma=0 and prediction is perfect"
        );
    }

    #[test]
    fn focal_loss_handles_batches_and_anchors() {
        let device = Default::default();

        // logits: [N=2, A=3, C=2]
        let logits = Tensor::<3>::from_floats(
            [
                [[-2.0, 2.0], [3.0, -3.0], [0.0, 0.0]],
                [[1.0, -1.0], [-4.0, 4.0], [2.0, -2.0]],
            ],
            &device,
        );

        let targets =
            Tensor::<2, Int>::from_ints([[1, 0, 1], [0, 1, 0]], &device);

        let loss = sigmoid_focal_loss(logits, targets, 0.25, 2.0);

        assert_eq!(loss.dims(), [1], "loss must reduce to scalar");
        assert!(scalar_value(loss).is_finite(), "loss must be finite");
    }

    #[test]
    fn focal_loss_gamma_effect_is_consistent() {
        let device = Default::default();

        let logits = Tensor::<3>::from_floats([[[0.0, 0.0]]], &device);
        let targets = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss_g0 = scalar_value(sigmoid_focal_loss(
            logits.clone(),
            targets.clone(),
            0.25,
            0.0,
        ));
        let loss_g2 = scalar_value(sigmoid_focal_loss(
            logits.clone(),
            targets.clone(),
            0.25,
            2.0,
        ));

        // for ambiguous predictions, gamma reduces the loss
        assert!(
            loss_g2 <= loss_g0,
            "gamma should reduce loss for ambiguous predictions"
        );
    }

    #[test]
    fn ssd_reg_zero_when_pred_equals_target() {
        let device = Default::default();

        let pred = Tensor::<3>::from_floats([[[0.0, 0.0, 0.0, 0.0]]], &device);
        let target =
            Tensor::<3>::from_floats([[[0.0, 0.0, 0.0, 0.0]]], &device);
        let mask = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = ssd_regression_loss(pred, target, mask);
        let value = scalar_value(loss);

        assert!(
            value < 1e-6,
            "loss should be ~0 when pred == target, got {}",
            value
        );
    }

    #[test]
    fn ssd_reg_ignores_negative_anchors() {
        let device = Default::default();

        let pred =
            Tensor::<3>::from_floats([[[10.0, -10.0, 5.0, -5.0]]], &device);
        let target =
            Tensor::<3>::from_floats([[[0.0, 0.0, 0.0, 0.0]]], &device);
        let mask = Tensor::<2, Int>::from_ints([[0]], &device);

        let loss = ssd_regression_loss(pred, target, mask);
        let value = scalar_value(loss);

        assert!(
            value < 1e-6,
            "loss should be ~0 when mask is zero, got {}",
            value
        );
    }

    #[test]
    fn ssd_reg_behaves_reasonably_for_small_errors() {
        let device = Default::default();

        let pred =
            Tensor::<3>::from_floats([[[0.1, -0.2, 0.3, -0.4]]], &device);
        let target = Tensor::<3>::zeros([1, 1, 4], &device);
        let mask = Tensor::<2, Int>::from_ints([[1]], &device);

        let loss = scalar_value(ssd_regression_loss(pred, target, mask));

        assert!(
            loss > 0.0 && loss < 1.0,
            "loss should be small and positive, got {}",
            loss
        );
    }

    #[test]
    fn ssd_reg_handles_batches_and_anchors() {
        let device = Default::default();

        let pred = Tensor::<3>::from_floats(
            [
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                [[-1.0, -1.0, -1.0, -1.0], [2.0, 2.0, 2.0, 2.0]],
            ],
            &device,
        ); // [2,2,4]

        let target = Tensor::<3>::zeros([2, 2, 4], &device);

        let mask = Tensor::<2, Int>::from_ints([[1, 0], [1, 1]], &device); // [2,2]

        let loss = ssd_regression_loss(pred, target, mask);
        let value = scalar_value(loss);

        assert!(value.is_finite(), "loss must be finite");
        assert!(
            value > 0.0,
            "loss should be positive when some anchors are active"
        );
    }
}
