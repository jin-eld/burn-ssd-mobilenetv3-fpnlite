use super::activation::{Activation, Relu6};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct ConvBNActivationConfig {
    pub in_planes: usize,
    pub out_planes: usize,

    #[config(default = "3")]
    pub kernel_size: usize,

    #[config(default = "1")]
    pub stride: usize,

    #[config(default = "1")]
    pub groups: usize,

    #[config(default = "1")]
    pub dilation: usize,

    #[config(default = "Activation::Relu6(Relu6::new())")]
    pub activation: Activation,
}

impl ConvBNActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBNActivation<B> {
        let padding = (self.kernel_size - 1) / 2 * self.dilation;

        return ConvBNActivation {
            conv: Conv2dConfig::new(
                [self.in_planes, self.out_planes],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .with_dilation([self.dilation, self.dilation])
            .with_groups(self.groups)
            .with_bias(false)
            .init(device),
            bn: BatchNormConfig::new(self.out_planes)
                .with_epsilon(0.001) // PyTorch defaults
                .with_momentum(0.01)
                .init(device),
            activation: self.activation.clone(),
        };
    }
}

#[derive(Module, Debug)]
pub struct ConvBNActivation<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    activation: Activation,
}

impl<B: Backend> ConvBNActivation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        return self.activation.forward(x);
    }
}
