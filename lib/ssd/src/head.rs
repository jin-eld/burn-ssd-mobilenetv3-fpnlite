use burn::{
    module::Module,
    tensor::{Device, Tensor},
};

use fpnlite::block::DepthwiseSeparableBlock;

#[derive(Module, Debug)]
pub struct SSDLiteHead {
    cls_heads: Vec<DepthwiseSeparableBlock>,
    bbox_heads: Vec<DepthwiseSeparableBlock>,
    num_anchors_per_level: Vec<usize>,
    num_classes: usize,
}

impl SSDLiteHead {
    pub fn new(
        in_channels: &[usize],           // one per FPN level
        num_anchors_per_level: &[usize], // one per FPN level
        num_classes: usize,
        device: &Device,
    ) -> Self {
        assert_eq!(
            in_channels.len(),
            num_anchors_per_level.len(),
            "in_channels and num_anchors_per_level must have same length"
        );

        let mut cls_heads = Vec::new();
        let mut bbox_heads = Vec::new();

        for (c, &a) in in_channels.iter().zip(num_anchors_per_level.iter()) {
            // classification: A * C channels
            cls_heads.push(DepthwiseSeparableBlock::new(
                *c,
                a * num_classes,
                device,
            ));

            // regression: A * 4 channels
            bbox_heads.push(DepthwiseSeparableBlock::new(*c, a * 4, device));
        }

        return Self {
            cls_heads,
            bbox_heads,
            num_anchors_per_level: num_anchors_per_level.to_vec(),
            num_classes,
        };
    }

    pub fn forward(
        &self,
        feats: &[Tensor<4>], // [N, C, H, W] per FPN level
    ) -> (Tensor<3>, Tensor<3>) {
        let mut cls_all = Vec::new();
        let mut bbox_all = Vec::new();

        for (i, feat) in feats.iter().enumerate() {
            let a = self.num_anchors_per_level[i];

            let cls = self.cls_heads[i].forward(feat.clone());
            let bbox = self.bbox_heads[i].forward(feat.clone());

            let shape = cls.shape();
            let dims = shape.as_slice();
            let n = dims[0];
            let h = dims[2];
            let w = dims[3];

            // cls: [N, A*C, H, W] -> [N, H*W*A, C]
            let cls = cls
                .reshape([n, a, self.num_classes, h, w])
                .permute([0, 3, 4, 1, 2])
                .reshape([n, h * w * a, self.num_classes]);

            // bbox: [N, A*4, H, W] -> [N, H*W*A, 4]
            let bbox = bbox
                .reshape([n, a, 4, h, w])
                .permute([0, 3, 4, 1, 2])
                .reshape([n, h * w * a, 4]);

            cls_all.push(cls);
            bbox_all.push(bbox);
        }

        let cls = Tensor::cat(cls_all, 1);
        let bbox = Tensor::cat(bbox_all, 1);

        return (cls, bbox);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn test_ssdlite_head_shapes() {
        let device = Device::default();

        let in_channels = [96, 96, 96, 96];
        let num_anchors_per_level = [3, 6, 6, 6]; // TFLite-style
        let num_classes = 91;

        let head = SSDLiteHead::new(
            &in_channels,
            &num_anchors_per_level,
            num_classes,
            &device,
        );

        let feats = [
            Tensor::<4>::zeros([1, 96, 20, 20], &device),
            Tensor::<4>::zeros([1, 96, 10, 10], &device),
            Tensor::<4>::zeros([1, 96, 5, 5], &device),
            Tensor::<4>::zeros([1, 96, 3, 3], &device),
        ];

        let (cls, bbox) = head.forward(&feats);

        let total_anchors = 20 * 20 * 3 + 10 * 10 * 6 + 5 * 5 * 6 + 3 * 3 * 6;

        assert_eq!(cls.shape().as_slice(), [1, total_anchors, num_classes]);
        assert_eq!(bbox.shape().as_slice(), [1, total_anchors, 4]);
    }
}
