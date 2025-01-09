use super::util::make_divisible;
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        HardSigmoid, HardSigmoidConfig, PaddingConfig2d, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct SqueezeExcitationConfig {
    input_channels: usize,

    #[config(default = "4")]
    squeeze_factor: usize,
}

impl SqueezeExcitationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SqueezeExcitation<B> {
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
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device),
            relu: Relu::new(),
            fc2: Conv2dConfig::new(
                [squeeze_channels, self.input_channels],
                [1, 1],
            )
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            hardsigmoid: HardSigmoidConfig::new().init(),
        };
    }
}

#[derive(Module, Debug)]
pub struct SqueezeExcitation<B: Backend> {
    fc1: Conv2d<B>,
    relu: Relu,
    fc2: Conv2d<B>,
    avgpool: AdaptiveAvgPool2d,
    hardsigmoid: HardSigmoid,
}

impl<B: Backend> SqueezeExcitation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let scale = self.avgpool.forward(input.clone());
        let scale = self.fc1.forward(scale);
        let scale = self.relu.forward(scale);
        let scale = self.fc2.forward(scale);
        return self.hardsigmoid.forward(scale) * input;
    }
}
