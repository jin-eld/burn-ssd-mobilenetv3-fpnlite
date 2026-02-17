//! Post‑processing for SSD‑Lite: softmax, filtering, per‑class NMS.

use crate::ops::nms::NmsOps;
use burn::prelude::Int;
use burn::tensor::activation::softmax;
use burn::tensor::Tensor;

/// A final detection after post‑processing.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class: usize,
}

fn tensor_to_vec_f32<const D: usize>(
    tensor: &Tensor<D>,
) -> Result<Vec<f32>, String> {
    return tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("Failed to extract f32 tensor data: {e}"));
}

fn tensor_to_vec_i32<const D: usize>(
    tensor: &Tensor<D, Int>,
) -> Result<Vec<i32>, String> {
    return tensor
        .clone()
        .into_data()
        .to_vec::<i32>()
        .map_err(|e| format!("Failed to extract i32 tensor data: {e}"));
}

/// SSD‑Lite post‑processing:
/// - softmax over classes
/// - remove background class
/// - pick best class per anchor
/// - filter by score threshold
/// - run per‑class NMS
pub fn ssd_postprocess(
    boxes: &Tensor<2>,        // [A, 4] decoded boxes (cx,cy,w,h)
    class_logits: &Tensor<2>, // [A, C] raw logits
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, String> {
    let num_anchors = boxes.dims()[0];
    let num_classes = class_logits.dims()[1];

    // Softmax over class dimension
    let probs = softmax(class_logits.clone(), 1);

    // Remove background class (index 0)
    let probs_no_bg = probs.slice([0..num_anchors, 1..num_classes]);

    // Best class per anchor
    let scores = probs_no_bg.clone().max_dim(1);
    let labels = probs_no_bg.argmax(1);

    // Move to CPU for filtering
    let scores_vec: Vec<f32> = tensor_to_vec_f32(&scores)?;
    let labels_i32: Vec<i32> = tensor_to_vec_i32(&labels)?;
    let labels_vec: Vec<usize> =
        labels_i32.iter().map(|v| *v as usize).collect();
    let boxes_data: Vec<f32> = tensor_to_vec_f32(&boxes)?;

    let mut boxes_vec = Vec::with_capacity(num_anchors);
    for i in 0..num_anchors {
        let base = i * 4;
        boxes_vec.push([
            boxes_data[base],
            boxes_data[base + 1],
            boxes_data[base + 2],
            boxes_data[base + 3],
        ]);
    }

    // Filter by score threshold
    let mut filtered_indices = Vec::new();
    for i in 0..num_anchors {
        if scores_vec[i] >= score_threshold {
            filtered_indices.push(i);
        }
    }

    // Group by class
    let mut detections = Vec::new();

    for class in 0..(num_classes - 1) {
        // Collect indices for this class
        let mut class_indices = Vec::new();
        for &i in &filtered_indices {
            if labels_vec[i] as usize == class {
                class_indices.push(i);
            }
        }

        if class_indices.is_empty() {
            continue;
        }

        // Build tensors for NMS
        let mut class_boxes = Vec::new();
        let mut class_scores = Vec::new();

        for &i in &class_indices {
            class_boxes.push(boxes_vec[i]);
            class_scores.push(scores_vec[i]);
        }

        // Convert to tensors
        let device = boxes.device();
        let flat_boxes: Vec<f32> =
            class_boxes.iter().flatten().copied().collect();

        let boxes_t = Tensor::<1>::from_floats(flat_boxes.as_slice(), &device)
            .reshape([class_boxes.len(), 4]);

        let scores_t =
            Tensor::<1>::from_floats(class_scores.as_slice(), &device);

        // Run NMS
        let keep = NmsOps::nms_single_class(
            &boxes_t,
            &scores_t,
            iou_threshold,
            max_detections,
        );

        // Collect detections
        for &k in &keep {
            detections.push(Detection {
                bbox: class_boxes[k],
                score: class_scores[k],
                class,
            });
        }
    }

    return Ok(detections);
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn test_postprocess_basic_softmax_and_filtering() {
        let device = Device::default();

        // 3 anchors, 3 classes (background + 2 real classes)
        let logits = Tensor::<2>::from_floats(
            [
                [0.0, 5.0, 1.0], // anchor 0 → class 1
                [0.0, 0.1, 0.2], // anchor 1 → class 2 (but low score)
                [0.0, 3.0, 0.1], // anchor 2 → class 1
            ],
            &device,
        );

        // Boxes (cx,cy,w,h)
        let boxes = Tensor::<2>::from_floats(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.7, 0.7, 0.2, 0.2],
            ],
            &device,
        );

        let detections = ssd_postprocess(
            &boxes, &logits, 0.5, // score threshold
            0.5, // IoU threshold
            10,  // max detections
        )
        .unwrap();

        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class, 0); // class 1 (background removed)
        assert_eq!(detections[1].class, 0);
    }

    #[test]
    fn test_postprocess_per_class_nms() {
        let device = Device::default();

        // Two anchors of class 1 that overlap heavily
        let logits = Tensor::<2>::from_floats(
            [
                [0.0, 5.0, 0.1], // class 1
                [0.0, 4.0, 0.1], // class 1
                [0.0, 0.1, 5.0], // class 2
            ],
            &device,
        );

        let boxes = Tensor::<2>::from_floats(
            [
                [0.5, 0.5, 0.4, 0.4], // overlaps with box 1
                [0.52, 0.52, 0.4, 0.4],
                [0.1, 0.1, 0.2, 0.2], // separate class
            ],
            &device,
        );

        let detections =
            ssd_postprocess(&boxes, &logits, 0.1, 0.5, 10).unwrap();

        // class 1 → only highest score kept
        // class 2 → kept
        assert_eq!(detections.len(), 2);

        let classes: Vec<usize> = detections.iter().map(|d| d.class).collect();
        assert!(classes.contains(&0)); // class 1
        assert!(classes.contains(&1)); // class 2
    }

    #[test]
    fn test_postprocess_score_threshold() {
        let device = Device::default();

        let logits = Tensor::<2>::from_floats(
            [
                [0.0, 0.1, 0.2], // low scores
                [0.0, 5.0, 0.1], // high score
            ],
            &device,
        );

        let boxes = Tensor::<2>::from_floats(
            [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
            &device,
        );

        let detections = ssd_postprocess(
            &boxes, &logits, 0.5, // threshold
            0.5, 10,
        )
        .unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].bbox, [0.5, 0.5, 0.2, 0.2]);
    }

    #[test]
    fn test_postprocess_max_detections() {
        let device = Device::default();

        let logits = Tensor::<2>::from_floats(
            [[0.0, 5.0, 0.1], [0.0, 4.0, 0.1], [0.0, 3.0, 0.1]],
            &device,
        );

        let boxes = Tensor::<2>::from_floats(
            [
                [0.1, 0.1, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.2],
            ],
            &device,
        );

        let detections = ssd_postprocess(
            &boxes, &logits, 0.1, 0.0, 2, // limit
        )
        .unwrap();

        assert_eq!(detections.len(), 2);
    }
}
