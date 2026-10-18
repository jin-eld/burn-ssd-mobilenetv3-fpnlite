//! COCO-style mean Average Precision (mAP) evaluation.
//!
//! Implements the standard COCO detection protocol:
//!   * detections are matched to ground truths greedily by descending score,
//!   * a match requires IoU >= threshold and each GT is claimed at most once,
//!   * Average Precision is the area under the Precision-Recall curve using
//!     COCO's 101-point interpolation,
//!   * mAP is averaged over all IoU thresholds (0.50..=0.95) and all classes
//!     that have at least one ground truth.
//!
//! Detections are expected to be pre-filtered (e.g. already NMS'd and
//! top-k limited), which is what the inference `postprocess` path produces.

use super::iou::iou_matrix;

/// COCO IoU thresholds: 0.50, 0.55, ..., 0.95.
pub const COCO_IOU_THRESHOLDS: [f32; 10] =
    [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

/// Number of recall points for COCO's interpolation (0.00, 0.01, ..., 1.00).
const NUM_RECALL_POINTS: usize = 101;

/// A single detection produced by the model (post-NMS).
#[derive(Clone, Debug)]
pub struct Detection {
    pub score: f32,
    pub class_id: usize,
    pub box_cxcywh: [f32; 4],
}

/// A single ground-truth annotation for one image.
#[derive(Clone, Debug)]
pub struct GroundTruth {
    pub class_id: usize,
    pub box_cxcywh: [f32; 4],
}

/// Aggregated mAP results.
#[derive(Clone, Debug)]
pub struct MapResult {
    /// mAP averaged over all classes (with GT) and all 10 IoU thresholds.
    pub map: f32,
    /// mAP at IoU = 0.50.
    pub map50: f32,
    /// mAP at IoU = 0.75.
    pub map75: f32,
    /// Per-class AP (mean over thresholds), indexed by class id. Classes
    /// without ground truth are excluded from `map` and left at 0.0 here.
    pub per_class_ap: Vec<f32>,
}

/// Compute COCO mAP over a set of images.
///
/// `predictions[i]` and `ground_truths[i]` are the detections and ground
/// truths for image `i`. Class ids must be in `0..num_classes`.
pub fn evaluate_map(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruth>],
    num_classes: usize,
) -> MapResult {
    assert_eq!(
        predictions.len(),
        ground_truths.len(),
        "predictions and ground truths must be per-image aligned"
    );

    // Track which classes actually have ground truth; classes without any
    // GT are excluded from the mean (COCO convention).
    let mut has_gt = vec![false; num_classes];
    for gts in ground_truths {
        for gt in gts {
            if gt.class_id < num_classes {
                has_gt[gt.class_id] = true;
            }
        }
    }

    // ap[class][threshold_index]
    let mut ap = vec![vec![0.0f32; COCO_IOU_THRESHOLDS.len()]; num_classes];
    for (t_idx, &t) in COCO_IOU_THRESHOLDS.iter().enumerate() {
        for c in 0..num_classes {
            if !has_gt[c] {
                continue;
            }
            ap[c][t_idx] =
                compute_ap_for_class(predictions, ground_truths, c, t);
        }
    }

    // Average each class's AP over all thresholds.
    let mut per_class_ap = vec![0.0f32; num_classes];
    let mut class_count = 0usize;
    for c in 0..num_classes {
        if !has_gt[c] {
            continue;
        }
        let sum: f32 = ap[c].iter().sum();
        per_class_ap[c] = sum / COCO_IOU_THRESHOLDS.len() as f32;
        class_count += 1;
    }

    let map = mean_over_classes(&per_class_ap, &has_gt, class_count);
    let map50 = mean_ap_at_threshold(&ap, &has_gt, 0);
    let map75 = mean_ap_at_threshold(&ap, &has_gt, 5);

    return MapResult {
        map,
        map50,
        map75,
        per_class_ap,
    };
}

/// Average the per-class AP values over classes that have ground truth.
fn mean_over_classes(
    per_class_ap: &[f32],
    has_gt: &[bool],
    class_count: usize,
) -> f32 {
    if class_count == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (c, &present) in has_gt.iter().enumerate() {
        if present {
            sum += per_class_ap[c];
        }
    }
    return sum / class_count as f32;
}

/// Average AP over classes (with GT) at a single threshold column.
fn mean_ap_at_threshold(ap: &[Vec<f32>], has_gt: &[bool], t_idx: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (c, &present) in has_gt.iter().enumerate() {
        if present {
            sum += ap[c][t_idx];
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    return sum / count as f32;
}

/// Compute the Average Precision for one class at one IoU threshold.
fn compute_ap_for_class(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruth>],
    class_id: usize,
    iou_threshold: f32,
) -> f32 {
    // Total ground truths for this class across all images.
    let total_gt: usize = ground_truths
        .iter()
        .map(|gts| gts.iter().filter(|g| g.class_id == class_id).count())
        .sum();
    if total_gt == 0 {
        return 0.0;
    }

    // Collect this class's detections as (image_idx, score, box).
    let mut dets: Vec<(usize, f32, [f32; 4])> = Vec::new();
    for (img_idx, preds) in predictions.iter().enumerate() {
        for p in preds {
            if p.class_id == class_id {
                dets.push((img_idx, p.score, p.box_cxcywh));
            }
        }
    }
    if dets.is_empty() {
        return 0.0;
    }

    // Process detections from highest to lowest score.
    dets.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Per-image, class-filtered ground-truth boxes.
    let gt_boxes_per_image: Vec<Vec<[f32; 4]>> = ground_truths
        .iter()
        .map(|gts| {
            gts.iter()
                .filter(|g| g.class_id == class_id)
                .map(|g| g.box_cxcywh)
                .collect()
        })
        .collect();

    let mut matched: Vec<Vec<bool>> = gt_boxes_per_image
        .iter()
        .map(|b| vec![false; b.len()])
        .collect();

    let mut cum_tp = 0.0f32;
    let mut cum_fp = 0.0f32;
    let mut precisions: Vec<f32> = Vec::with_capacity(dets.len());
    let mut recalls: Vec<f32> = Vec::with_capacity(dets.len());

    for (img_idx, _score, det_box) in &dets {
        let gt_boxes = &gt_boxes_per_image[*img_idx];

        // Find the best-IoU unmatched GT in this image that clears the
        // threshold.
        let mut best_gt: Option<usize> = None;
        let mut best_iou = iou_threshold;
        if !gt_boxes.is_empty() {
            let ious = iou_matrix(&[*det_box], gt_boxes); // shape [1][G]
            for (g, &iou_val) in ious[0].iter().enumerate() {
                if !matched[*img_idx][g] && iou_val >= best_iou {
                    best_iou = iou_val;
                    best_gt = Some(g);
                }
            }
        }

        if let Some(g) = best_gt {
            matched[*img_idx][g] = true;
            cum_tp += 1.0;
        } else {
            cum_fp += 1.0;
        }

        precisions.push(cum_tp / (cum_tp + cum_fp));
        recalls.push(cum_tp / total_gt as f32);
    }

    return interpolate_ap(&precisions, &recalls);
}

/// COCO's 101-point interpolation of the Precision-Recall curve.
///
/// At each of the 101 recall levels, take the maximum precision observed
/// at any recall >= that level, then average across all levels.
fn interpolate_ap(precisions: &[f32], recalls: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for k in 0..NUM_RECALL_POINTS {
        let r = k as f32 / (NUM_RECALL_POINTS - 1) as f32;
        let mut max_precision = 0.0f32;
        for i in 0..precisions.len() {
            if recalls[i] >= r && precisions[i] > max_precision {
                max_precision = precisions[i];
            }
        }
        sum += max_precision;
    }
    return sum / NUM_RECALL_POINTS as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt(class_id: usize, box_cxcywh: [f32; 4]) -> GroundTruth {
        return GroundTruth {
            class_id,
            box_cxcywh,
        };
    }

    fn det(score: f32, class_id: usize, box_cxcywh: [f32; 4]) -> Detection {
        return Detection {
            score,
            class_id,
            box_cxcywh,
        };
    }

    const EPS: f32 = 1e-6;

    #[test]
    fn test_perfect_detection_map_is_one() {
        let preds = vec![vec![det(0.9, 0, [0.5, 0.5, 0.4, 0.4])]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 1);
        assert!((res.map - 1.0).abs() < EPS);
        assert!((res.map50 - 1.0).abs() < EPS);
        assert!((res.map75 - 1.0).abs() < EPS);
    }

    #[test]
    fn test_no_detections_map_is_zero() {
        let preds: Vec<Vec<Detection>> = vec![vec![]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 1);
        assert!((res.map - 0.0).abs() < EPS);
        assert!((res.map50 - 0.0).abs() < EPS);
    }

    #[test]
    fn test_high_score_false_positive_halves_ap() {
        // A high-score false positive is processed before the lower-score
        // true positive, capping the precision envelope at 0.5.
        let preds = vec![vec![
            det(0.9, 0, [0.1, 0.1, 0.1, 0.1]), // no overlap -> FP
            det(0.8, 0, [0.5, 0.5, 0.4, 0.4]), // exact -> TP
        ]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 1);
        assert!((res.map - 0.5).abs() < EPS);
        assert!((res.map50 - 0.5).abs() < EPS);
    }

    #[test]
    fn test_shifted_box_passes_iou50_fails_iou75() {
        // GT [0.5,0.5,0.4,0.4] -> xyxy [0.3,0.3,0.7,0.7].
        // Detection center shifted +0.12 in x -> IoU ~= 0.538.
        // TP at threshold 0.50, FP at every threshold >= 0.55.
        let preds = vec![vec![det(0.9, 0, [0.62, 0.5, 0.4, 0.4])]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 1);
        assert!((res.map50 - 1.0).abs() < EPS);
        assert!((res.map75 - 0.0).abs() < EPS);
        // Only 1 of the 10 thresholds is a true positive -> mAP = 0.1.
        assert!((res.map - 0.1).abs() < EPS);
    }

    #[test]
    fn test_class_without_gt_is_excluded() {
        // Class 0 has a perfect detection; class 1 has a detection but no
        // GT, so it must not drag the mean down.
        let preds = vec![vec![
            det(0.9, 0, [0.5, 0.5, 0.4, 0.4]),
            det(0.9, 1, [0.2, 0.2, 0.1, 0.1]),
        ]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 2);
        assert!((res.map - 1.0).abs() < EPS);
        assert!((res.per_class_ap[0] - 1.0).abs() < EPS);
        // Class 1 has no GT -> excluded from the mean, slot stays 0.0.
        assert!((res.per_class_ap[1] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_duplicate_detection_second_is_fp() {
        // Two identical detections for one GT: higher score is the TP, the
        // second is an FP. Recall still reaches 1.0, so the envelope keeps
        // AP at 1.0 (standard COCO behaviour).
        let preds = vec![vec![
            det(0.9, 0, [0.5, 0.5, 0.4, 0.4]),
            det(0.8, 0, [0.5, 0.5, 0.4, 0.4]),
        ]];
        let gts = vec![vec![gt(0, [0.5, 0.5, 0.4, 0.4])]];
        let res = evaluate_map(&preds, &gts, 1);
        assert!((res.map - 1.0).abs() < EPS);
    }

    #[test]
    fn test_multiple_images_partial_recall() {
        // Two images, class 0. Image 0's GT is detected, image 1's GT is
        // missed. Recall = 1/2 with precision 1.0. With 101-point
        // interpolation, recall 0.5 covers levels 0.00..=0.50 (51 points).
        let preds = vec![vec![det(0.9, 0, [0.5, 0.5, 0.4, 0.4])], vec![]];
        let gts = vec![
            vec![gt(0, [0.5, 0.5, 0.4, 0.4])],
            vec![gt(0, [0.5, 0.5, 0.4, 0.4])],
        ];
        let res = evaluate_map(&preds, &gts, 1);
        let expected = 51.0 / 101.0;
        assert!((res.map - expected).abs() < 1e-4);
        assert!((res.map50 - expected).abs() < 1e-4);
    }
}
