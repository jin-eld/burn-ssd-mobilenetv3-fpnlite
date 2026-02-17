use crate::block::DepthwiseSeparableBlock;
use burn::config::Config;
use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2dConfig, InterpolateMode},
    },
    tensor::{Device, Tensor},
};
use mobilenetv3::MobileNetV3Arch;

#[derive(Config, Debug)]
pub struct FpnLiteConfig {
    pub out_channels: usize,
    pub backbone: MobileNetV3Arch,
}

#[derive(Module, Debug)]
pub struct FpnLite {
    // lateral 1×1 projections
    lateral_c3: Conv2d,
    lateral_c4: Conv2d,

    // smoothing blocks
    smooth_p3: DepthwiseSeparableBlock,
    smooth_p4: DepthwiseSeparableBlock,
    smooth_p5: DepthwiseSeparableBlock,
    smooth_p6: DepthwiseSeparableBlock,

    // downsampling (stride‑2 depthwise)
    downsample_p4: DepthwiseSeparableBlock,
    downsample_p5: DepthwiseSeparableBlock,
}

impl FpnLite {
    pub fn new(cfg: FpnLiteConfig, device: &Device) -> Self {
        let out = cfg.out_channels;

        let (c3_in, c4_in) = match cfg.backbone {
            MobileNetV3Arch::Large => (80, 160),
            MobileNetV3Arch::Small => (40, 96),
        };

        return Self {
            lateral_c3: Conv2dConfig::new([c3_in, out], [1, 1]).init(device),
            lateral_c4: Conv2dConfig::new([c4_in, out], [1, 1]).init(device),

            smooth_p3: DepthwiseSeparableBlock::new(out, out, device),
            smooth_p4: DepthwiseSeparableBlock::new(out, out, device),
            smooth_p5: DepthwiseSeparableBlock::new(out, out, device),
            smooth_p6: DepthwiseSeparableBlock::new(out, out, device),

            downsample_p4: DepthwiseSeparableBlock::new_stride2(
                out, out, device,
            ),
            downsample_p5: DepthwiseSeparableBlock::new_stride2(
                out, out, device,
            ),
        };
    }

    pub fn forward(&self, c3: Tensor<4>, c4: Tensor<4>) -> [Tensor<4>; 4] {
        let l3 = self.lateral_c3.forward(c3);
        let l4 = self.lateral_c4.forward(c4);

        let target_h = l3.shape()[2];
        let target_w = l3.shape()[3];

        let upsample = Interpolate2dConfig::new()
            .with_output_size(Some([target_h, target_w]))
            .with_mode(InterpolateMode::Nearest)
            .init();

        let up4 = upsample.forward(l4.clone());

        let p3 = self.smooth_p3.forward(l3 + up4);
        let p4 = self.smooth_p4.forward(l4);
        let p5 = self
            .smooth_p5
            .forward(self.downsample_p4.forward(p4.clone()));
        let p6 = self
            .smooth_p6
            .forward(self.downsample_p5.forward(p5.clone()));

        return [p3, p4, p5, p6];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};
    use mobilenetv3::{MobileNetV3Arch, MobileNetV3Config};

    #[test]
    fn test_fpnlite_shapes_large() {
        let device = Device::default();

        let config = MobileNetV3Config::new();
        let backbone = config.init_large(&device); // large variant

        let cfg = FpnLiteConfig::new(96, MobileNetV3Arch::Large);
        let fpnlite = FpnLite::new(cfg, &device);

        // fake input image
        let x = Tensor::<4>::zeros([1, 3, 320, 320], &device);

        let (c3, c4) = backbone.forward_features(x);

        // run FPNLite
        let [p3, p4, p5, p6] = fpnlite.forward(c3, c4);

        assert_eq!(p3.shape().as_slice(), [1, 96, 20, 20]);
        assert_eq!(p4.shape().as_slice(), [1, 96, 10, 10]);
        assert_eq!(p5.shape().as_slice(), [1, 96, 5, 5]);
        assert_eq!(p6.shape().as_slice(), [1, 96, 3, 3]);
    }

    #[test]
    fn test_fpnlite_shapes_small() {
        let device = Device::default();

        let config = MobileNetV3Config::new();
        let backbone = config.init_small(&device); // small variant

        let cfg = FpnLiteConfig::new(96, MobileNetV3Arch::Small);
        let fpnlite = FpnLite::new(cfg, &device);

        // fake input image
        let x = Tensor::<4>::zeros([1, 3, 320, 320], &device);

        let (c3, c4) = backbone.forward_features(x);

        let [p3, p4, p5, p6] = fpnlite.forward(c3, c4);

        // expected shapes
        assert_eq!(p3.shape().as_slice(), [1, 96, 20, 20]);
        assert_eq!(p4.shape().as_slice(), [1, 96, 10, 10]);
        assert_eq!(p5.shape().as_slice(), [1, 96, 5, 5]);
        assert_eq!(p6.shape().as_slice(), [1, 96, 3, 3]);
    }

    #[test]
    fn test_fpnlite_backbone_channel_consistency() {
        use burn::tensor::{Device, Tensor};
        use mobilenetv3::{MobileNetV3Arch, MobileNetV3Config};

        let device = Device::default();
        let config = MobileNetV3Config::new();

        // Helper closure to extract (c3_channels, c4_channels)
        let extract_channels = |arch| {
            let backbone = match arch {
                MobileNetV3Arch::Large => config.init_large(&device),
                MobileNetV3Arch::Small => config.init_small(&device),
            };

            let x = Tensor::<4>::zeros([1, 3, 320, 320], &device);
            let (c3, c4) = backbone.forward_features(x);

            (c3.shape().as_slice()[1], c4.shape().as_slice()[1])
        };

        // Extract actual backbone outputs
        let (c3_large, c4_large) = extract_channels(MobileNetV3Arch::Large);
        let (c3_small, c4_small) = extract_channels(MobileNetV3Arch::Small);

        // Expected values used by FpnLiteConfig
        let expected_large = (80, 160);
        let expected_small = (40, 96);

        assert_eq!(
            (c3_large, c4_large),
            expected_large,
            "MobileNetV3-Large C3/C4 channels changed — update FpnLiteConfig!"
        );

        assert_eq!(
            (c3_small, c4_small),
            expected_small,
            "MobileNetV3-Small C3/C4 channels changed — update FpnLiteConfig!"
        );
    }
}
