use crate::ops::iou::iou_matrix;

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
pub struct Matcher {
    pub positive_iou_threshold: f32,
    pub negative_iou_threshold: f32,
}

impl Matcher {
    pub fn new(pos: f32, neg: f32) -> Self {
        Self {
            positive_iou_threshold: pos,
            negative_iou_threshold: neg,
        }
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

        // Edge case: no ground truth → all background, zero deltas
        if num_gt == 0 {
            return (vec![0; num_anchors], vec![[0.0; 4]; num_anchors]);
        }

        // 1. IoU matrix [A, G]
        let ious = iou_matrix(anchors, gt_boxes);

        // 2. For each anchor, find best GT
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

        // 3. Initialize labels: -1 = ignore
        let mut labels = vec![-1i32; num_anchors];

        for a in 0..num_anchors {
            let iou = best_iou_for_anchor[a];
            if iou >= self.positive_iou_threshold {
                labels[a] = gt_labels[best_gt_for_anchor[a]] as i32;
            } else if iou < self.negative_iou_threshold {
                labels[a] = 0; // background
            }
        }

        // 4. Ensure each GT has at least one positive anchor
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

        // 5. Encode regression targets for positive anchors
        let mut encoded = vec![[0.0; 4]; num_anchors];

        for a in 0..num_anchors {
            if labels[a] > 0 {
                let g = best_gt_for_anchor[a];
                encoded[a] = encode_ssd_box(anchors[a], gt_boxes[g]);
            }
        }

        (labels, encoded)
    }
}

/// SSD box encoding:
/// tx = (gx - ax) / aw
/// ty = (gy - ay) / ah
/// tw = log(gw / aw)
/// th = log(gh / ah)
fn encode_ssd_box(anchor: [f32; 4], gt: [f32; 4]) -> [f32; 4] {
    let (ax, ay, aw, ah) = (anchor[0], anchor[1], anchor[2], anchor[3]);
    let (gx, gy, gw, gh) = (gt[0], gt[1], gt[2], gt[3]);

    [
        (gx - ax) / aw,
        (gy - ay) / ah,
        (gw / aw).ln(),
        (gh / ah).ln(),
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

        // Even with IoU=0, SSD forces a positive match for each GT
        assert_eq!(labels[0], 1);
    }

    #[test]
    fn test_matcher_ensures_each_gt_has_positive() {
        // Two anchors, one GT; one anchor overlaps more
        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];
        let gt_boxes = vec![[0.5, 0.5, 0.4, 0.4]];
        let gt_labels = vec![2];

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, _encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        // At least one anchor must be positive with label 2
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
        // Two anchors far away from the GT box → IoU = 0 for both
        let anchors = vec![[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]];

        let gt_boxes = vec![
            [0.9, 0.9, 0.1, 0.1], // completely non-overlapping
        ];

        let gt_labels = vec![5]; // arbitrary class

        let matcher = Matcher::new(0.5, 0.4);
        let (labels, _encoded) =
            matcher.match_anchors(&anchors, &gt_boxes, &gt_labels);

        // At least one anchor must be assigned class 5
        assert!(
            labels.iter().any(|&l| l == 5),
            "Matcher must force a positive match even when IoU = 0"
        );

        // The other anchor should be background (0) or ignore (-1),
        // depending on thresholds — both are valid SSD behavior.
        let positives = labels.iter().filter(|&&l| l == 5).count();
        assert_eq!(
            positives, 1,
            "Exactly one anchor should be forced positive"
        );
    }
}
