use super::activation::{Activation, Hardswish};
use super::conv_bn_activation::{ConvBNActivation, ConvBNActivationConfig};
use super::identity::IdentityConfig;
use super::squeeze_excitation::{SqueezeExcitation, SqueezeExcitationConfig};
use super::util::adjust_channels;
use burn::{
    config::Config,
    module::Module,
    nn::Relu,
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum BottleneckActivationType {
    Hardswish,
    Relu,
}

#[derive(Config, Debug)]
pub struct InvertedResidualConfig {
    pub input_channels: usize,
    pub kernel: usize,
    pub expanded_channels: usize,
    pub out_channels: usize,
    pub use_se: bool,
    pub activation_type: BottleneckActivationType,
    pub stride: usize,
    pub dilation: usize,
    pub width_mult: f64,
}

impl InvertedResidualConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidual<B> {
        if self.stride < 1 || self.stride > 2 {
            panic!("illegal stride value");
        }

        let input_channels =
            adjust_channels(self.input_channels, self.width_mult);
        let expanded_channels =
            adjust_channels(self.expanded_channels, self.width_mult);
        let out_channels = adjust_channels(self.out_channels, self.width_mult);

        let expand_conv = if expanded_channels != input_channels {
            Some(
                ConvBNActivationConfig::new(input_channels, expanded_channels)
                    .with_kernel_size(1)
                    .with_activation(match self.activation_type {
                        BottleneckActivationType::Hardswish => {
                            Activation::Hardswish(Hardswish::new())
                        }
                        BottleneckActivationType::Relu => {
                            Activation::Relu(Relu::new())
                        }
                    })
                    .init(device),
            )
        } else {
            None
        };

        let stride = if self.dilation > 1 { 1 } else { self.stride };
        let depthwise_conv =
            ConvBNActivationConfig::new(expanded_channels, expanded_channels)
                .with_kernel_size(self.kernel)
                .with_stride(stride)
                .with_groups(expanded_channels)
                .with_dilation(self.dilation)
                .with_activation(match self.activation_type {
                    BottleneckActivationType::Hardswish => {
                        Activation::Hardswish(Hardswish::new())
                    }
                    BottleneckActivationType::Relu => {
                        Activation::Relu(Relu::new())
                    }
                })
                .init(device);

        let se_layer = if self.use_se {
            Some(SqueezeExcitationConfig::new(expanded_channels).init(device))
        } else {
            None
        };

        let project_conv =
            ConvBNActivationConfig::new(expanded_channels, out_channels)
                .with_kernel_size(1)
                .with_activation(Activation::Identity(
                    IdentityConfig::new().init(),
                ))
                .init(device);

        return InvertedResidual {
            expand_conv,
            depthwise_conv,
            se_layer,
            project_conv,
            use_res_connect: self.stride == 1 && input_channels == out_channels,
        };
    }
}

#[derive(Module, Debug)]
pub struct InvertedResidual<B: Backend> {
    expand_conv: Option<ConvBNActivation<B>>,
    depthwise_conv: ConvBNActivation<B>,
    se_layer: Option<SqueezeExcitation<B>>,
    project_conv: ConvBNActivation<B>,
    use_res_connect: bool,
}

impl<B: Backend> InvertedResidual<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = input.clone();

        if let Some(ref conv) = self.expand_conv {
            x = conv.forward(x);
        }

        x = self.depthwise_conv.forward(x);

        if let Some(ref se) = self.se_layer {
            x = se.forward(x);
        }

        x = self.project_conv.forward(x);

        if self.use_res_connect {
            x = x + input;
        }

        return x;
    }
}
