use burn::{
    module::Module,
    tensor::{Device, Tensor},
};

use fpnlite::{FpnLite, FpnLiteConfig};
use mobilenetv3::{MobileNetV3, MobileNetV3Arch, MobileNetV3Config};

use crate::ops::postprocess::{ssd_postprocess, Detection};
use crate::{AnchorGenerator, BoxDecoder, DecodeConfig, SSDLiteHead};

#[derive(Debug, Module)]
pub struct SSDLiteMobileNetV3Model {
    backbone: MobileNetV3,
    fpn: FpnLite,
    head: SSDLiteHead,
}

#[derive(Debug, Module)]
pub struct SSDLiteMobileNetV3 {
    model: SSDLiteMobileNetV3Model,
    #[module(skip)]
    anchors: AnchorGenerator,
    #[module(skip)]
    decoder: BoxDecoder,
    #[module(skip)]
    score_threshold: f32,
    #[module(skip)]
    iou_threshold: f32,
    #[module(skip)]
    max_detections: usize,
}

impl SSDLiteMobileNetV3Model {
    /// Forward pass up to the SSD head (no anchors, no decoding).
    ///
    /// Returns:
    /// - cls_logits:  [N, A, num_classes]
    /// - bbox_deltas: [N, A, 4]
    pub fn forward_head(&self, x: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
        // 1. backbone: get C3 and C4 feature maps
        let (c3, c4) = self.backbone.forward_features(x);

        // 2. FPNLite: get P3–P6 feature maps
        let feats = self.fpn.forward(c3, c4);

        // 3. SSDLite head: get class logits and box deltas
        return self.head.forward(&feats);
    }
}

impl SSDLiteMobileNetV3 {
    pub fn new(
        arch: MobileNetV3Arch,
        num_classes: usize,
        device: &Device,
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
            anchors,
            decoder,
            score_threshold: 0.5,
            iou_threshold: 0.5,
            max_detections: 200,
        };
    }

    pub fn inner_model(&self) -> &SSDLiteMobileNetV3Model {
        &self.model
    }

    pub fn decoder(&self) -> &BoxDecoder {
        &self.decoder
    }

    /// Full forward pass: image -> backbone -> FPN -> head -> decoded boxes.
    ///
    /// Returns:
    /// - cls_logits: [N, A, num_classes]
    /// - boxes:      [N, A, 4] (cx, cy, w, h) in normalized coords
    pub fn forward_raw(&self, x: Tensor<4>) -> (Tensor<3>, Tensor<3>) {
        return self.model.forward_head(x);
    }

    pub fn forward(&self, input: Tensor<4>) -> Result<Vec<Detection>, String> {
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
        device: &Device,
    ) -> Vec<[f32; 4]> {
        // dummy input
        let dummy = Tensor::<4>::zeros(
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
        self.anchors.generate(&feature_map_sizes)
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{MobileNetV3Arch, SSDLiteMobileNetV3};
    use burn::tensor::{Device, Tensor};

    #[test]
    fn test_ssd_end_to_end_forward() {
        let device = Device::default();

        let model = SSDLiteMobileNetV3::new(MobileNetV3Arch::Small, 3, &device);

        let input = Tensor::<4>::zeros([1, 3, 160, 160], &device);

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
