use super::util::make_divisible;
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        HardSigmoid, HardSigmoidConfig, PaddingConfig2d, Relu,
    },
    tensor::{Device, Tensor},
};

#[derive(Config, Debug)]
pub struct SqueezeExcitationConfig {
    pub input_channels: usize,

    #[config(default = "4")]
    pub squeeze_factor: usize,
}

impl SqueezeExcitationConfig {
    pub fn init(&self, device: &Device) -> SqueezeExcitation {
        let squeeze_channels = make_divisible(
            self.input_channels as f64 / self.squeeze_factor as f64,
            8,
            None,
        );

        return SqueezeExcitation {
            fc1: Conv2dConfig::new(
                [self.input_channels, squeeze_channels],
                [1, 1],
            )
            .with_padding(PaddingConfig2d::Valid)
            .init(device),
            relu: Relu::new(),
            fc2: Conv2dConfig::new(
                [squeeze_channels, self.input_channels],
                [1, 1],
            )
            .with_padding(PaddingConfig2d::Valid)
            .init(device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            hardsigmoid: HardSigmoidConfig::new()
                .with_alpha(1_f64 / 6_f64) // match PyTorch implementation
                .init(),
        };
    }
}

#[derive(Module, Debug)]
pub struct SqueezeExcitation {
    fc1: Conv2d,
    relu: Relu,
    fc2: Conv2d,
    avgpool: AdaptiveAvgPool2d,
    hardsigmoid: HardSigmoid,
}

impl SqueezeExcitation {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.avgpool.forward(input.clone());
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        return self.hardsigmoid.forward(x) * input;
    }
}
