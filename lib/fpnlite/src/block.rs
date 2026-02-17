use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

use mobilenetv3::Relu6;

#[derive(Module, Debug)]
pub struct DepthwiseSeparableBlock<B: Backend> {
    depthwise: Conv2d<B>,
    pointwise: Conv2d<B>,
    activation: Relu6,
}

impl<B: Backend> DepthwiseSeparableBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> Self {
        return Self::build(in_channels, out_channels, 1, device);
    }

    pub fn new_stride2(
        in_channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> Self {
        return Self::build(in_channels, out_channels, 2, device);
    }

    fn build(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        let depthwise = Conv2dConfig::new([in_channels, in_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_groups(in_channels)
            .init(device);

        let pointwise = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);

        return Self {
            depthwise,
            pointwise,
            activation: Relu6::new(),
        };
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(x);
        let x = self.pointwise.forward(x);
        return self.activation.forward(x);
    }
}
