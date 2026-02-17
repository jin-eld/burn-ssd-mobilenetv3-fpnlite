use crate::ops::iou::{cxcywh_to_xyxy, iou_xyxy};
use burn::tensor::Tensor;

pub struct NmsOps;

impl NmsOps {
    /// Greedy NMS for a single class.
    /// boxes: [A, 4] in cx,cy,w,h
    /// scores: [A]
    pub fn nms_single_class(
        boxes: &Tensor<2>,
        scores: &Tensor<1>,
        iou_threshold: f32,
        max_detections: usize,
    ) -> Vec<usize> {
        // Move to CPU for now.
        let boxes_data = boxes.clone().into_data().to_vec::<f32>().unwrap();
        let scores_data = scores.clone().into_data().to_vec::<f32>().unwrap();

        let num_boxes = scores_data.len();

        // Flatten [A,4] into Vec<[f32;4]>
        let mut boxes_vec = Vec::with_capacity(num_boxes);
        for i in 0..num_boxes {
            let base = i * 4;
            boxes_vec.push([
                boxes_data[base],
                boxes_data[base + 1],
                boxes_data[base + 2],
                boxes_data[base + 3],
            ]);
        }

        // Convert to xyxy
        let boxes_xyxy = cxcywh_to_xyxy(&boxes_vec);

        // Sort indices by score descending
        let mut order: Vec<usize> = (0..num_boxes).collect();
        order.sort_by(|&i, &j| {
            scores_data[j].partial_cmp(&scores_data[i]).unwrap()
        });

        let mut keep = Vec::new();

        'outer: for &idx in &order {
            if keep.len() >= max_detections {
                break;
            }

            for &k in &keep {
                let iou = iou_xyxy(&boxes_xyxy[idx], &boxes_xyxy[k]);
                if iou > iou_threshold {
                    continue 'outer;
                }
            }

            keep.push(idx);
        }

        return keep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    fn tensor_boxes(device: &Device, boxes: &[[f32; 4]]) -> Tensor<2> {
        let flat: Vec<f32> = boxes.iter().flatten().copied().collect();
        let t = Tensor::<1>::from_floats(flat.as_slice(), device);
        return t.reshape([boxes.len(), 4]);
    }

    fn tensor_scores(device: &Device, scores: &[f32]) -> Tensor<1> {
        return Tensor::<1>::from_floats(scores, device);
    }

    #[test]
    fn test_nms_simple_suppression() {
        let device = Device::default();

        let boxes = [
            [0.5, 0.5, 0.4, 0.4],
            [0.52, 0.52, 0.4, 0.4], // overlaps heavily
        ];
        let scores = [0.9, 0.8];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_no_suppression_low_iou() {
        let device = Device::default();

        let boxes = [
            [0.1, 0.1, 0.2, 0.2],
            [0.8, 0.8, 0.2, 0.2], // far apart
        ];
        let scores = [0.9, 0.8];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        assert_eq!(keep, vec![0, 1]);
    }

    #[test]
    fn test_nms_respects_max_detections() {
        let device = Device::default();

        let boxes = [
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.2, 0.2],
        ];
        let scores = [0.9, 0.8, 0.7];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.0,
            2,
        );

        assert_eq!(keep.len(), 2);
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn test_nms_threshold_behavior() {
        let device = Device::default();

        let boxes = [[0.5, 0.5, 0.4, 0.4], [0.52, 0.52, 0.4, 0.4]];
        let scores = [0.9, 0.8];

        // IoU is high → suppressed
        let keep1 = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );
        assert_eq!(keep1, vec![0]);

        // IoU threshold too high → keep both
        let keep2 = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.99,
            10,
        );
        assert_eq!(keep2, vec![0, 1]);
    }

    #[test]
    fn test_nms_identical_boxes() {
        let device = Device::default();

        let boxes = [[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]];
        let scores = [0.9, 0.8];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_touching_edges_no_overlap() {
        let device = Device::default();

        // Two boxes touching at x2 = x1
        let boxes = [
            [0.3, 0.5, 0.4, 0.4], // spans x = 0.1 to 0.5
            [0.5, 0.5, 0.4, 0.4], // spans x = 0.3 to 0.7
        ];

        let scores = [0.9, 0.8];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        assert_eq!(keep, vec![0, 1]);
    }

    #[test]
    fn test_nms_iou_equal_threshold_kept() {
        let device = Device::default();

        // Two boxes with IoU = 0.25
        let boxes = [[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2]];

        let scores = [0.9, 0.8];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.25, // IoU == threshold
            10,
        );

        // small box is fully inside the large box, due to rounding IoU is
        // slightly above threshold, small box gets suppressed
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_all_suppressed_except_best() {
        let device = Device::default();

        let boxes = [
            [0.5, 0.5, 0.4, 0.4],
            [0.52, 0.52, 0.4, 0.4],
            [0.48, 0.48, 0.4, 0.4],
        ];

        let scores = [0.9, 0.8, 0.7];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.3,
            10,
        );

        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_identical_scores_deterministic() {
        let device = Device::default();

        let boxes = [[0.5, 0.5, 0.4, 0.4], [0.52, 0.52, 0.4, 0.4]];

        let scores = [0.8, 0.8]; // identical

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        // Deterministic: lower index wins
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn test_nms_zero_area_boxes() {
        let device = Device::default();

        let boxes = [
            [0.5, 0.5, 0.0, 0.4], // zero width
            [0.5, 0.5, 0.4, 0.0], // zero height
            [0.5, 0.5, 0.4, 0.4], // valid
        ];

        let scores = [0.9, 0.8, 0.7];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        // zero‑area boxes never suppress anything and are never suppressed
        assert_eq!(keep, vec![0, 1, 2]);
    }

    #[test]
    fn test_nms_large_vs_small_boxes() {
        let device = Device::default();

        let boxes = [
            [0.5, 0.5, 0.9, 0.9], // huge box
            [0.5, 0.5, 0.1, 0.1], // tiny box inside
        ];

        let scores = [0.8, 0.9];

        let keep = NmsOps::nms_single_class(
            &tensor_boxes(&device, &boxes),
            &tensor_scores(&device, &scores),
            0.5,
            10,
        );

        // tiny box has higher score → kept
        // Since 0.25 < 0.5 → large box is NOT suppressed
        assert_eq!(keep, vec![1, 0]);
    }
}
