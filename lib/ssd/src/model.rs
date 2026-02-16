use burn::{
    module::{Ignored, Module},
    tensor::{backend::Backend, Tensor},
};

use fpnlite::{FpnLite, FpnLiteConfig};
use mobilenetv3::{MobileNetV3, MobileNetV3Arch, MobileNetV3Config};

use crate::ops::postprocess::{ssd_postprocess, Detection};
use crate::{AnchorGenerator, BoxDecoder, DecodeConfig, SSDLiteHead};

#[derive(Debug, Module)]
pub struct SSDLiteMobileNetV3Model<B: Backend> {
    backbone: MobileNetV3<B>,
    fpn: FpnLite<B>,
    head: SSDLiteHead<B>,
}

#[derive(Debug, Module)]
pub struct SSDLiteMobileNetV3<B: Backend> {
    model: SSDLiteMobileNetV3Model<B>,
    anchors: Ignored<AnchorGenerator>,
    decoder: Ignored<BoxDecoder>,
    score_threshold: Ignored<f32>,
    iou_threshold: Ignored<f32>,
    max_detections: Ignored<usize>,
}

impl<B: Backend> SSDLiteMobileNetV3Model<B> {
    /// Forward pass up to the SSD head (no anchors, no decoding).
    ///
    /// Returns:
    /// - cls_logits:  [N, A, num_classes]
    /// - bbox_deltas: [N, A, 4]
    pub fn forward_head(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // 1. backbone: get C3 and C4 feature maps
        let (c3, c4) = self.backbone.forward_features(x);

        // 2. FPNLite: get P3–P6 feature maps
        let feats = self.fpn.forward(c3, c4);

        // 3. SSDLite head: get class logits and box deltas
        return self.head.forward(&feats);
    }
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

        // anchor generator (SSD-style, matching num_anchors_per_level)
        let anchors = AnchorGenerator::new(
            0.2,                                 // min_scale (TFLite-ish)
            0.95,                                // max_scale
            vec![1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], // aspect ratios
            true,      // clip to [0,1], TFLite clip is always on
            true,      // reduce_boxes_in_lowest_layer
            Some(1.0), // interpolated_scale_aspect_ratio
        );

        let decoder = BoxDecoder::new(DecodeConfig::default());

        return Self {
            model,
            anchors: Ignored(anchors),
            decoder: Ignored(decoder),
            score_threshold: Ignored(0.5),
            iou_threshold: Ignored(0.5),
            max_detections: Ignored(200),
        };
    }

    pub fn inner_model(&self) -> &SSDLiteMobileNetV3Model<B> {
        &self.model
    }

    pub fn decoder(&self) -> &BoxDecoder {
        &self.decoder.0
    }

    /// Full forward pass: image -> backbone -> FPN -> head -> decoded boxes.
    ///
    /// Returns:
    /// - cls_logits: [N, A, num_classes]
    /// - boxes:      [N, A, 4] (cx, cy, w, h) in normalized coords
    pub fn forward_raw(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        return self.model.forward_head(x);
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
            self.score_threshold.0,
            self.iou_threshold.0,
            self.max_detections.0,
        );
    }

    /// Training helper: generate anchors for a given input size by running a
    /// dummy forward pass through the backbone + FPN to obtain the true
    /// feature map sizes.
    ///
    /// This guarantees that anchors match the SSDLite head exactly.
    // TODO: was a pure testing helper, but now we need it elsewhere too
    // -> rather training feature    #[cfg(any(test, feature = "test-utils"))]
    pub fn generate_anchors_for_input(
        &self,
        input_h: u32,
        input_w: u32,
        device: &B::Device,
    ) -> Vec<[f32; 4]> {
        // dummy input
        let dummy = Tensor::<B, 4>::zeros(
            [1, 3, input_h as usize, input_w as usize],
            device,
        );

        // backbone -> C3, C4
        let (c3, c4) = self.model.backbone.forward_features(dummy);

        // FPN -> P3–P6
        let feats = self.model.fpn.forward(c3, c4);

        // extract feature map sizes
        let feature_map_sizes: Vec<(usize, usize)> = feats
            .iter()
            .map(|f| {
                let d = f.dims();
                (d[2], d[3]) // (H, W)
            })
            .collect();

        // generate anchors using the model’s AnchorGenerator
        self.anchors.0.generate(&feature_map_sizes)
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{MobileNetV3Arch, SSDLiteMobileNetV3};
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;

    #[test]
    fn test_ssd_end_to_end_forward() {
        let device = WgpuDevice::default();

        let model =
            SSDLiteMobileNetV3::<Wgpu>::new(MobileNetV3Arch::Small, 3, &device);

        let input = Tensor::<Wgpu, 4>::zeros([1, 3, 160, 160], &device);

        let detections = model
            .forward(input)
            .expect("SSD forward pass should not fail");

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
