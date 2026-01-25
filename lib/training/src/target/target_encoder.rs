use crate::target::matcher::Matcher;
use burn::tensor::{backend::Backend, DType, Int, Tensor, TensorData};

#[derive(Clone, Debug)]
pub struct SSDTargetEncoder {
    matcher: Matcher,
}

impl SSDTargetEncoder {
    pub fn new(matcher: Matcher) -> Self {
        return Self { matcher };
    }

    pub fn encode_batch<B: Backend>(
        &self,
        anchors: &[[f32; 4]],
        batch_gt_boxes: Vec<Vec<[f32; 4]>>,
        batch_gt_labels: Vec<Vec<usize>>,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 3>, Tensor<B, 2, Int>) {
        let batch_size = batch_gt_boxes.len();
        let num_anchors = anchors.len();

        let mut all_labels: Vec<i32> =
            Vec::with_capacity(batch_size * num_anchors);
        let mut all_boxes: Vec<f32> =
            Vec::with_capacity(batch_size * num_anchors * 4);
        let mut all_pos_mask: Vec<i32> =
            Vec::with_capacity(batch_size * num_anchors);

        for i in 0..batch_size {
            let (labels, encoded) = self.matcher.match_anchors(
                anchors,
                &batch_gt_boxes[i],
                &batch_gt_labels[i],
            );

            for &l in &labels {
                all_labels.push(l);
                all_pos_mask.push(if l > 0 { 1 } else { 0 });
            }

            for e in &encoded {
                all_boxes.extend_from_slice(e); // push 4 floats
            }
        }

        let labels_data = TensorData::from_bytes_vec(
            all_labels.iter().flat_map(|x| x.to_le_bytes()).collect(),
            [batch_size, num_anchors],
            DType::I32,
        );

        let pos_mask_data = TensorData::from_bytes_vec(
            all_pos_mask.iter().flat_map(|x| x.to_le_bytes()).collect(),
            [batch_size, num_anchors],
            DType::I32,
        );

        let boxes_data = TensorData::from_bytes_vec(
            all_boxes.iter().flat_map(|x| x.to_le_bytes()).collect(),
            [batch_size, num_anchors, 4],
            DType::F32,
        );

        let tgt_classes = Tensor::<B, 2, Int>::from_data(labels_data, device);
        let pos_mask = Tensor::<B, 2, Int>::from_data(pos_mask_data, device);
        let tgt_boxes = Tensor::<B, 3>::from_data(boxes_data, device);

        return (tgt_classes, tgt_boxes, pos_mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{Int, Tensor};

    type B = Wgpu;

    fn to_vec_i32(t: &Tensor<B, 2, Int>) -> Vec<Vec<i32>> {
        let data = t.to_data();
        let flat = data.to_vec::<i32>().unwrap();
        let shape = t.shape();
        let cols = shape.dims[1];

        flat.chunks(cols).map(|c| c.to_vec()).collect()
    }

    fn to_vec_f32_3d(t: &Tensor<B, 3>) -> Vec<Vec<[f32; 4]>> {
        let data = t.to_data();
        let flat = data.to_vec::<f32>().unwrap();
        let shape = t.shape();
        let batch = shape.dims[0];
        let anchors = shape.dims[1];

        let mut out = Vec::new();
        let mut idx = 0;

        for _ in 0..batch {
            let mut rows = Vec::new();
            for _ in 0..anchors {
                let slice =
                    [flat[idx], flat[idx + 1], flat[idx + 2], flat[idx + 3]];
                rows.push(slice);
                idx += 4;
            }
            out.push(rows);
        }
        out
    }

    #[test]
    fn test_encoder_single_image_single_gt() {
        let device = Default::default();

        let anchors = vec![[0.5, 0.5, 0.4, 0.4]];
        let gt_boxes = vec![vec![[0.5, 0.5, 0.4, 0.4]]];
        let gt_labels = vec![vec![3]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);
        let boxes = to_vec_f32_3d(&tgt_boxes);

        assert_eq!(classes, vec![vec![3]]);
        assert_eq!(mask, vec![vec![1]]);

        let b = boxes[0][0];
        assert!(b[0].abs() < 1e-6);
        assert!(b[1].abs() < 1e-6);
        assert!(b[2].abs() < 1e-6);
        assert!(b[3].abs() < 1e-6);
    }

    #[test]
    fn test_encoder_single_image_no_gt() {
        let device = Default::default();

        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];
        let gt_boxes = vec![vec![]];
        let gt_labels = vec![vec![]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);
        let boxes = to_vec_f32_3d(&tgt_boxes);

        assert_eq!(classes, vec![vec![0, 0]]);
        assert_eq!(mask, vec![vec![0, 0]]);

        for b in &boxes[0] {
            assert!(b.iter().all(|v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn test_encoder_batch_two_images() {
        let device = Default::default();

        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];

        let gt_boxes = vec![
            vec![[0.5, 0.5, 0.4, 0.4]], // image 0
            vec![[0.2, 0.2, 0.1, 0.1]], // image 1
        ];

        let gt_labels = vec![vec![1], vec![2]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, _tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);

        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].len(), 2);
        assert_eq!(classes[1].len(), 2);

        assert!(classes[0].contains(&1));
        assert!(classes[1].contains(&2));

        assert!(mask[0].iter().any(|&m| m == 1));
        assert!(mask[1].iter().any(|&m| m == 1));
    }

    #[test]
    fn test_encoder_multiple_gt_in_single_image() {
        let device = Default::default();

        // Two anchors, two GT boxes
        let anchors = vec![
            [0.5, 0.5, 0.4, 0.4], // overlaps GT0
            [0.2, 0.2, 0.1, 0.1], // overlaps GT1
        ];

        let gt_boxes = vec![vec![
            [0.5, 0.5, 0.4, 0.4], // class 1
            [0.2, 0.2, 0.1, 0.1], // class 2
        ]];

        let gt_labels = vec![vec![1, 2]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);
        let boxes = to_vec_f32_3d(&tgt_boxes);

        assert_eq!(classes[0], vec![1, 2]);
        assert_eq!(mask[0], vec![1, 1]);

        // both anchors perfectly match -> zero deltas
        for encoded in &boxes[0] {
            assert!(encoded.iter().all(|v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn test_encoder_forced_positive_even_with_zero_iou() {
        let device = Default::default();

        let anchors = vec![[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1]];

        let gt_boxes = vec![vec![
            [0.9, 0.9, 0.1, 0.1], // no overlap with any anchor
        ]];

        let gt_labels = vec![vec![7]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, _tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);

        // exactly one forced positive
        assert_eq!(classes[0].iter().filter(|&&c| c == 7).count(), 1);
        assert_eq!(mask[0].iter().filter(|&&m| m == 1).count(), 1);
    }

    #[test]
    fn test_encoder_encoded_deltas_correctness() {
        let device = Default::default();

        let anchors = vec![[0.5, 0.5, 0.4, 0.4]];

        let gt_boxes = vec![vec![
            [0.6, 0.55, 0.5, 0.3], // slightly offset and different size
        ]];

        let gt_labels = vec![vec![4]];

        let matcher = Matcher::new(0.0, 0.0); // force positive
        let encoder = SSDTargetEncoder::new(matcher);

        let (_tgt_classes, tgt_boxes, _pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let boxes = to_vec_f32_3d(&tgt_boxes);
        let e = boxes[0][0];

        // expected encoding:
        let tx = (0.6 - 0.5) / 0.4;
        let ty = (0.55 - 0.5) / 0.4;
        let tw = (0.5f32 / 0.4).ln();
        let th = (0.3f32 / 0.4).ln();

        assert!((e[0] - tx).abs() < 1e-6);
        assert!((e[1] - ty).abs() < 1e-6);
        assert!((e[2] - tw).abs() < 1e-6);
        assert!((e[3] - th).abs() < 1e-6);
    }

    #[test]
    fn test_encoder_batch_mixed_gt_counts() {
        let device = Default::default();

        let anchors = vec![[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]];

        let gt_boxes = vec![
            vec![[0.5, 0.5, 0.4, 0.4]], // image 0: 1 GT
            vec![],                     // image 1: no GT
        ];

        let gt_labels = vec![vec![1], vec![]];

        let matcher = Matcher::new(0.5, 0.4);
        let encoder = SSDTargetEncoder::new(matcher);

        let (tgt_classes, _tgt_boxes, pos_mask) =
            encoder.encode_batch::<B>(&anchors, gt_boxes, gt_labels, &device);

        let classes = to_vec_i32(&tgt_classes);
        let mask = to_vec_i32(&pos_mask);

        // image 0: one positive
        assert!(classes[0].contains(&1));
        assert!(mask[0].iter().any(|&m| m == 1));

        // image 1: all background
        assert_eq!(classes[1], vec![0, 0]);
        assert_eq!(mask[1], vec![0, 0]);
    }
}
