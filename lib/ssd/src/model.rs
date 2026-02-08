use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use fpnlite::{FpnLite, FpnLiteConfig};
use mobilenetv3::{MobileNetV3, MobileNetV3Arch, MobileNetV3Config};

use crate::ops::postprocess::{ssd_postprocess, Detection};
use crate::{AnchorGenerator, BoxDecoder, DecodeConfig, SSDLiteHead};

#[derive(Module, Debug)]
pub struct SSDLiteMobileNetV3Model<B: Backend> {
    backbone: MobileNetV3<B>,
    fpn: FpnLite<B>,
    head: SSDLiteHead<B>,
}

#[derive(Debug)]
pub struct SSDLiteMobileNetV3<B: Backend> {
    model: SSDLiteMobileNetV3Model<B>,
    anchors: AnchorGenerator,
    decoder: BoxDecoder,
    score_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
}

impl<B: Backend> SSDLiteMobileNetV3<B> {
    pub fn new(
        arch: MobileNetV3Arch,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        // Backbone
        let backbone_cfg = MobileNetV3Config::new();
        let backbone = match arch {
            MobileNetV3Arch::Large => backbone_cfg.init_large(device),
            MobileNetV3Arch::Small => backbone_cfg.init_small(device),
        };

        // FPNLite
        let fpn_cfg = FpnLiteConfig::new(96, arch);
        let fpn = FpnLite::new(fpn_cfg, device);

        // SSDLite head
        let in_channels = vec![96, 96, 96, 96]; // P3–P6
        let num_anchors_per_level = vec![3, 6, 6, 6]; // TFLite-style
        let head = SSDLiteHead::new(
            &in_channels,
            &num_anchors_per_level,
            num_classes,
            device,
        );

        let model = SSDLiteMobileNetV3Model {
            backbone,
            fpn,
            head,
        };

        // Anchor generator (SSD-style, matching num_anchors_per_level)
        let anchors = AnchorGenerator::new(
            0.2,                                 // min_scale (TFLite-ish)
            0.95,                                // max_scale
            vec![1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], // aspect ratios
            true,      // clip to [0,1], TFLite clip is always on
            true,      // reduce_boxes_in_lowest_layer
            Some(1.0), // interpolated_scale_aspect_ratio
        );

        // Decoder
        let decoder = BoxDecoder::new(DecodeConfig::default());

        return Self {
            model,
            anchors,
            decoder,
            score_threshold: 0.5,
            iou_threshold: 0.5,
            max_detections: 200,
        };
    }

    /// Full forward pass: image → backbone → FPN → head → decoded boxes.
    ///
    /// Returns:
    /// - cls_logits: [N, A, num_classes]
    /// - boxes:      [N, A, 4] (cx, cy, w, h) in normalized coords
    fn forward_raw(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // 1. Backbone: get C3 and C4 feature maps
        let (c3, c4) = self.model.backbone.forward_features(x);

        // 2. FPNLite: get P3–P6 feature maps
        let feats = self.model.fpn.forward(c3, c4);

        let feature_map_sizes: Vec<(usize, usize)> = feats
            .iter()
            .map(|f| {
                let d = f.dims();
                (d[2], d[3]) // (H, W)
            })
            .collect();

        // 3. SSDLite head: get class logits and box deltas
        let (cls_logits, bbox_deltas) = self.model.head.forward(&feats);
        // cls_logits:  [N, A, C]
        // bbox_deltas: [N, A, 4]

        // 4. Anchors: generate and convert to tensor [A, 4]
        let anchors_vec = self.anchors.generate(&feature_map_sizes);
        let num_anchors = anchors_vec.len();

        // Flatten Vec<[f32; 4]> into Vec<f32>
        let flat: Vec<f32> =
            anchors_vec.iter().flat_map(|a| a.iter().copied()).collect();

        let device = cls_logits.device();

        // From_floats only takes (data, device), so we create a 1D tensor and
        // reshape it
        let anchors_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(flat.as_slice(), &device)
                .reshape([num_anchors, 4]);

        // 5. Decode boxes: [N, A, 4]
        let boxes = self
            .decoder
            .decode::<B>(bbox_deltas.clone(), anchors_tensor);

        return (cls_logits, boxes);
    }

    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Result<Vec<Detection>, String> {
        let (class_logits, decoded_boxes) = self.forward_raw(input);

        // Expect batch size 1
        let dims_logits = class_logits.dims();
        let dims_boxes = decoded_boxes.dims();

        assert_eq!(
            dims_logits[0], 1,
            "forward() currently supports batch size 1 only"
        );
        assert_eq!(
            dims_boxes[0], 1,
            "forward() currently supports batch size 1 only"
        );

        // Remove batch dimension 0
        let class_logits = class_logits.squeeze_dim::<2>(0);
        let decoded_boxes = decoded_boxes.squeeze_dim::<2>(0);

        return ssd_postprocess(
            &decoded_boxes,
            &class_logits,
            self.score_threshold,
            self.iou_threshold,
            self.max_detections,
        );
    }

    /// Training helper Generate anchors for a given input size by running a
    /// dummy forward pass through the backbone + FPN to obtain the true
    /// feature map sizes.
    ///
    /// This guarantees that anchors match the SSDLite head exactly.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn generate_anchors_for_input(
        &self,
        input_h: usize,
        input_w: usize,
        device: &B::Device,
    ) -> Vec<[f32; 4]> {
        // 1. Dummy input
        let dummy = Tensor::<B, 4>::zeros([1, 3, input_h, input_w], device);

        // 2. Backbone → C3, C4
        let (c3, c4) = self.model.backbone.forward_features(dummy);

        // 3. FPN → P3–P6
        let feats = self.model.fpn.forward(c3, c4);

        // 4. Extract feature map sizes
        let feature_map_sizes: Vec<(usize, usize)> = feats
            .iter()
            .map(|f| {
                let d = f.dims();
                (d[2], d[3]) // (H, W)
            })
            .collect();

        // 5. Generate anchors using the model’s AnchorGenerator
        self.anchors.generate(&feature_map_sizes)
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{MobileNetV3Arch, SSDLiteMobileNetV3};
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;

    #[test]
    fn test_ssd_end_to_end_forward() {
        // 1. Create device
        let device = WgpuDevice::default();

        // 2. Create model (MobileNetV3 Small or Large)
        let model = SSDLiteMobileNetV3::<Wgpu>::new(
            MobileNetV3Arch::Small,
            3, // num_classes (background + 2 classes)
            &device,
        );

        // 3. Dummy input image [1, 3, H, W]
        // Use a small resolution to keep test fast
        let input = Tensor::<Wgpu, 4>::zeros([1, 3, 160, 160], &device);

        // 4. Run full forward pass
        let detections = model
            .forward(input)
            .expect("SSD forward pass should not fail");

        // 5. Basic sanity checks
        // The model may output 0 detections depending on random weights,
        // so we only check structural correctness.
        for det in &detections {
            assert!(det.score >= 0.0 && det.score <= 1.0);
            assert!(det.class < 2); // since num_classes = 3 (background removed)
            assert_eq!(det.bbox.len(), 4);

            // bounding box sanity
            let [cx, cy, w, h] = det.bbox;
            assert!(cx >= 0.0 && cx <= 1.0);
            assert!(cy >= 0.0 && cy <= 1.0);
            assert!(w >= 0.0 && w <= 1.0);
            assert!(h >= 0.0 && h <= 1.0);
        }
    }

    #[test]
    fn test_ssd_decoded_boxes_are_normalized() {
        use crate::model::{MobileNetV3Arch, SSDLiteMobileNetV3};
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        use burn::tensor::Tensor;

        // 1. Device
        let device = WgpuDevice::default();

        // 2. Model with 3 classes
        let model =
            SSDLiteMobileNetV3::<Wgpu>::new(MobileNetV3Arch::Small, 3, &device);

        // 3. Dummy input [1, 3, H, W]
        let input = Tensor::<Wgpu, 4>::zeros([1, 3, 160, 160], &device);

        // 4. Run raw forward
        let (_cls_logits, boxes) = model.forward_raw(input);

        // 5. Expect [1, A, 4]
        let dims = boxes.dims();
        assert_eq!(dims.len(), 3);
        assert_eq!(dims[0], 1);
        assert_eq!(dims[2], 4);

        // 6. Remove batch dimension
        let boxes = boxes.squeeze_dim::<2>(0);

        // 7. Extract data
        let data = boxes
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract decoded box data");

        // 8. Validate each box
        for chunk in data.chunks(4) {
            let cx = chunk[0];
            let cy = chunk[1];
            let w = chunk[2];
            let h = chunk[3];

            // No NaNs or infinities
            assert!(cx.is_finite());
            assert!(cy.is_finite());
            assert!(w.is_finite());
            assert!(h.is_finite());

            // Normalized center coordinates
            assert!(cx >= 0.0 && cx <= 1.0);
            assert!(cy >= 0.0 && cy <= 1.0);

            // Width/height must be non-negative
            assert!(w >= 0.0);
            assert!(h >= 0.0);

            // Width/height should not exceed 1.0 in normalized space
            assert!(w <= 1.0);
            assert!(h <= 1.0);
        }
    }

    #[test]
    fn test_ssd_forward_multi_resolution() {
        let device = WgpuDevice::default();

        // Test several resolutions
        let resolutions = [160, 320, 640];

        for &size in &resolutions {
            let model = SSDLiteMobileNetV3::<Wgpu>::new(
                MobileNetV3Arch::Small,
                3, // num_classes
                &device,
            );

            // Dummy input
            let input = Tensor::<Wgpu, 4>::zeros([1, 3, size, size], &device);

            // Run full forward pass
            let detections = model
                .forward(input)
                .expect("SSD forward pass should not fail");

            // Structural checks
            for det in &detections {
                assert!(det.score >= 0.0 && det.score <= 1.0);
                assert!(det.class < 2);
                assert_eq!(det.bbox.len(), 4);

                let [cx, cy, w, h] = det.bbox;

                assert!(cx >= 0.0 && cx <= 1.0);
                assert!(cy >= 0.0 && cy <= 1.0);
                assert!(w >= 0.0 && w <= 1.0);
                assert!(h >= 0.0 && h <= 1.0);
            }
        }
    }

    #[test]
    fn test_anchor_and_prediction_count_match() {
        let device = WgpuDevice::default();
        let sizes = [160, 320, 640];

        for &size in &sizes {
            let model = SSDLiteMobileNetV3::<Wgpu>::new(
                MobileNetV3Arch::Small,
                3,
                &device,
            );

            let input = Tensor::<Wgpu, 4>::zeros([1, 3, size, size], &device);
            let (cls, boxes) = model.forward_raw(input);

            assert_eq!(
                cls.dims()[1],
                boxes.dims()[1],
                "Anchor/pred count mismatch at size {size}"
            );
        }
    }
}
