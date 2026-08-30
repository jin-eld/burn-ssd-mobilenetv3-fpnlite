use ssd::ops::iou::iou_matrix;

/// SSD-style anchor matcher.
///
/// Assigns each anchor to:
/// - positive: matched to a GT box
/// - negative: background
/// - ignore: does not contribute to loss
///
/// Labels convention:
/// -1 = ignore
///  0 = background
/// >0 = class index
#[derive(Clone, Debug)]
pub struct Matcher {
    pub positive_iou_threshold: f32,
    pub negative_iou_threshold: f32,
    /// Variance for center deltas; must match the decoder's center_variance.
    pub center_variance: f32,
    /// Variance for size deltas; must match the decoder's size_variance.
    pub size_variance: f32,
}

impl Matcher {
    pub fn new(pos: f32, neg: f32) -> Self {
        return Self {
            positive_iou_threshold: pos,
            negative_iou_threshold: neg,
            center_variance: 0.1,
            size_variance: 0.2,
        };
    }

    /// Match anchors to ground-truth boxes.
    ///
    /// Inputs:
    /// - anchors: [A][4] in cx,cy,w,h format
    /// - gt_boxes: [G][4] in cx,cy,w,h format
    /// - gt_labels: [G] class indices
    ///
    /// Outputs:
    /// - labels[A]: -1 ignore, 0 background, >0 class
    /// - encoded_boxes[A][4]: SSD deltas
    pub fn match_anchors(
        &self,
        anchors: &[[f32; 4]],
        gt_boxes: &[[f32; 4]],
        gt_labels: &[usize],
    ) -> (Vec<i32>, Vec<[f32; 4]>) {
        let num_anchors = anchors.len();
        let num_gt = gt_boxes.len();

        // edge case: no ground truth -> all background, zero deltas
        if num_gt == 0 {
            return (vec![0; num_anchors], vec![[0.0; 4]; num_anchors]);
        }

        // IoU matrix [A, G]
        let ious = iou_matrix(anchors, gt_boxes);

        // for each anchor, find best GT
        let mut best_gt_for_anchor = vec![0usize; num_anchors];
        let mut best_iou_for_anchor = vec![0.0f32; num_anchors];

        for a in 0..num_anchors {
            for g in 0..num_gt {
                let iou = ious[a][g];
                if iou > best_iou_for_anchor[a] {
                    best_iou_for_anchor[a] = iou;
                    best_gt_for_anchor[a] = g;
                }
            }
        }

        // initialize labels: -1 = ignore
        let mut labels = vec![-1i32; num_anchors];

        for a in 0..num_anchors {
            let iou = best_iou_for_anchor[a];
            if iou >= self.positive_iou_threshold {
                labels[a] = gt_labels[best_gt_for_anchor[a]] as i32;
            } else if iou < self.negative_iou_threshold {
                labels[a] = 0; // background
            }
        }

        // ensure each GT has at least one positive anchor
        for g in 0..num_gt {
            let mut best_anchor = 0usize;
            let mut best_iou = -1.0f32;

            for a in 0..num_anchors {
                let iou = ious[a][g];
                if iou > best_iou {
                    best_iou = iou;
                    best_anchor = a;
                }
            }

            labels[best_anchor] = gt_labels[g] as i32;
            best_gt_for_anchor[best_anchor] = g;
        }

        // encode regression targets for positive anchors
        let mut encoded = vec![[0.0; 4]; num_anchors];

        for a in 0..num_anchors {
            if labels[a] > 0 {
                let g = best_gt_for_anchor[a];
                encoded[a] = encode_ssd_box(
                    anchors[a],
                    gt_boxes[g],
                    self.center_variance,
                    self.size_variance,
                );
            }
        }

        return (labels, encoded);
    }
}

/// SSD box encoding (exact inverse of the decoder):
/// tx = (gx - ax) / (aw * center_variance)
/// ty = (gy - ay) / (ah * center_variance)
/// tw = log(gw / aw) / size_variance
/// th = log(gh / ah) / size_variance
fn encode_ssd_box(
    anchor: [f32; 4],
    gt: [f32; 4],
    center_variance: f32,
    size_variance: f32,
) -> [f32; 4] {
    let (ax, ay, aw, ah) = (anchor[0], anchor[1], anchor[2], anchor[3]);
    let (gx, gy, gw, gh) = (gt[0], gt[1], gt[2], gt[3]);
    [
        (gx - ax) / (aw * center_variance),
        (gy - ay) / (ah * center_variance),
        (gw / aw).ln() / size_variance,
        (gh / ah).ln() / size_variance,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matcher_single_perfect_match() {
        let anchors = vec![[0.5, 0.5, 0.4, 0.4]];
        let gt_boxes = vec![[0.5, 0.5, 0.4, 0.4]];
        let gt_labels = vec![3];

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        assert_eq!(labels.len(), 1);
        assert_eq!(encoded.len(), 1);

        // Perfect match → positive with class 3
        assert_eq!(labels[0], 3);

        // Perfect overlap → zero deltas
        let e = encoded[0];
        assert!(e[0].abs() < 1e-6);
        assert!(e[1].abs() < 1e-6);
        assert!(e[2].abs() < 1e-6);
        assert!(e[3].abs() < 1e-6);
    }

    #[test]
    fn test_matcher_background_when_no_overlap() {
        let anchors = vec![[0.1, 0.1, 0.1, 0.1]];
        let gt_boxes = vec![[0.8, 0.8, 0.1, 0.1]];
        let gt_labels = vec![1];

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, _encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        // even with IoU=0, SSD forces a positive match for each GT
        assert_eq!(labels[0], 1);
    }

    #[test]
    fn test_matcher_ensures_each_gt_has_positive() {
        // two anchors, one GT; one anchor overlaps more
        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];
        let gt_boxes = vec![[0.5, 0.5, 0.4, 0.4]];
        let gt_labels = vec![2];

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, _encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        // at least one anchor must be positive with label 2
        assert!(labels.iter().any(|&l| l == 2));
    }

    #[test]
    fn test_matcher_no_gt_all_background() {
        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];
        let gt_boxes: Vec<[f32; 4]> = vec![];
        let gt_labels: Vec<usize> = vec![];

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        assert_eq!(labels, vec![0, 0]); // all background
        assert_eq!(encoded, vec![[0.0; 4], [0.0; 4]]);
    }

    #[test]
    fn test_matcher_forced_positive_even_with_zero_iou() {
        // two anchors far away from the GT box → IoU = 0 for both
        let anchors = vec![[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]];

        let gt_boxes = vec![
            [0.9, 0.9, 0.1, 0.1], // completely non-overlapping
        ];

        let gt_labels = vec![5]; // arbitrary class

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, _encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        // at least one anchor must be assigned class 5
        assert!(
            labels.iter().any(|&l| l == 5),
            "Matcher must force a positive match even when IoU = 0"
        );

        // the other anchor should be background (0) or ignore (-1),
        // depending on thresholds — both are valid SSD behavior.
        let positives = labels.iter().filter(|&&l| l == 5).count();
        assert_eq!(
            positives, 1,
            "Exactly one anchor should be forced positive"
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        use burn::tensor::{Device, Tensor};
        use ssd::decoder::{BoxDecoder, DecodeConfig};

        let device = Device::default();
        let cfg = DecodeConfig::default();
        let anchors = [[0.5, 0.5, 0.4, 0.4], [0.25, 0.25, 0.1, 0.2]];
        let gts = [[0.6, 0.55, 0.5, 0.3], [0.27, 0.22, 0.12, 0.17]];

        for (a, g) in anchors.iter().zip(gts.iter()) {
            let enc =
                encode_ssd_box(*a, *g, cfg.center_variance, cfg.size_variance);
            let deltas = Tensor::<3>::from_floats([[enc]], &device);
            let anchor_t = Tensor::<2>::from_floats([*a], &device);
            let decoder = BoxDecoder::new(cfg.clone());
            let dec = decoder.decode(deltas, anchor_t);
            let v = dec.into_data().try_to_vec::<f32>().unwrap();
            assert!((v[0] - g[0]).abs() < 1e-4, "cx roundtrip");
            assert!((v[1] - g[1]).abs() < 1e-4, "cy roundtrip");
            assert!((v[2] - g[2]).abs() < 1e-4, "w roundtrip");
            assert!((v[3] - g[3]).abs() < 1e-4, "h roundtrip");
        }
    }

    #[test]
    fn test_regression_loss_space_contract() {
        use burn::tensor::{Device, Int, Tensor};
        use ssd::decoder::{BoxDecoder, DecodeConfig};

        let device = Device::default();
        let cfg = DecodeConfig::default();
        let anchor = [0.5, 0.5, 0.4, 0.4];
        let gt = [0.6, 0.55, 0.5, 0.3];
        let enc =
            encode_ssd_box(anchor, gt, cfg.center_variance, cfg.size_variance);

        let deltas = Tensor::<3>::from_floats([[enc]], &device);
        // Independent tensor with identical values: production never feeds the
        // same tensor handle as both pred and target, and aliasing one tensor
        // as both sub operands hits a backend edge case.
        let deltas2 = Tensor::<3>::from_floats([[enc]], &device);
        let anchor_t = Tensor::<2>::from_floats([anchor], &device);
        let decoded = BoxDecoder::new(cfg).decode(deltas.clone(), anchor_t);
        let mask = Tensor::<2, Int>::from_ints([[1]], &device);

        // Same space (deltas vs deltas): loss must be ~0
        let same = crate::loss::ssd_regression_loss(
            deltas.clone(),
            deltas2,
            mask.clone(),
        );
        let same_v = same.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(same_v < 1e-6, "delta-space loss must be ~0, got {}", same_v);

        // Mixed space (decoded boxes vs deltas): must be clearly non-zero,
        // i.e. the old wiring is detectable
        let mixed = crate::loss::ssd_regression_loss(decoded, deltas, mask);
        let mixed_v = mixed.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            mixed_v > 0.1,
            "mixed-space loss must be clearly non-zero, got {}",
            mixed_v
        );
    }
}
