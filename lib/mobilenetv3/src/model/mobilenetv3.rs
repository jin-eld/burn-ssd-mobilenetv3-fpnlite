use super::activation::{Activation, Hardswish};
use super::conv_bn_activation::{ConvBNActivation, ConvBNActivationConfig};
use super::inverted_residual::{
    BottleneckActivationType, InvertedResidual, InvertedResidualConfig,
};
use super::squeeze_excitation::SqueezeExcitation;
use super::util::*;
use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Device, Tensor},
};

#[cfg(feature = "pretrained")]
use {
    super::weights::{self, WeightsMeta},
    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

#[derive(Module, Debug)]
enum LayerType<B: Backend> {
    ConvBNActivation(ConvBNActivation<B>),
    SqueezeExcitation(SqueezeExcitation<B>),
    InvertedResidual(InvertedResidual<B>),
}

impl<B: Backend> LayerType<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::ConvBNActivation(block) => block.forward(x),
            Self::SqueezeExcitation(block) => block.forward(x),
            Self::InvertedResidual(block) => block.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct Classifier<B: Backend> {
    fc1: Linear<B>,
    activation: Hardswish,
    dropout: Dropout,
    fc2: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

pub enum MobileNetV3Arch {
    Large,
    Small,
}

fn mobilenet_v3_conf(
    arch: MobileNetV3Arch,
    width_mult: f64,
    reduced_tail: bool,
    dilated: bool,
) -> (Vec<InvertedResidualConfig>, usize) {
    let reduce_divider = if reduced_tail { 2 } else { 1 };
    let dilation = if dilated { 2 } else { 1 };

    let bneck_conf = |input_channels,
                      kernel,
                      expanded_channels,
                      out_channels,
                      use_se,
                      activation,
                      stride,
                      dilation| {
        InvertedResidualConfig::new(
            input_channels,
            kernel,
            expanded_channels,
            out_channels,
            use_se,
            activation,
            stride,
            dilation,
            width_mult,
        )
    };

    const RE: BottleneckActivationType = BottleneckActivationType::Relu;
    const HS: BottleneckActivationType = BottleneckActivationType::Hardswish;

    match arch {
        MobileNetV3Arch::Large => {
            let settings = vec![
                bneck_conf(16, 3, 16, 16, false, RE, 1, 1),
                bneck_conf(16, 3, 64, 24, false, RE, 2, 1),
                bneck_conf(24, 3, 72, 24, false, RE, 1, 1),
                bneck_conf(24, 5, 72, 40, true, RE, 2, 1),
                bneck_conf(40, 5, 120, 40, true, RE, 1, 1),
                bneck_conf(40, 5, 120, 40, true, RE, 1, 1),
                bneck_conf(40, 3, 240, 80, false, HS, 2, 1),
                bneck_conf(80, 3, 200, 80, false, HS, 1, 1),
                bneck_conf(80, 3, 184, 80, false, HS, 1, 1),
                bneck_conf(80, 3, 184, 80, false, HS, 1, 1),
                bneck_conf(80, 3, 480, 112, true, HS, 1, 1),
                bneck_conf(112, 3, 672, 112, true, HS, 1, 1),
                bneck_conf(
                    112,
                    5,
                    672,
                    160 / reduce_divider,
                    true,
                    HS,
                    2,
                    dilation,
                ),
                bneck_conf(
                    160 / reduce_divider,
                    5,
                    960 / reduce_divider,
                    160 / reduce_divider,
                    true,
                    HS,
                    1,
                    dilation,
                ),
                bneck_conf(
                    160 / reduce_divider,
                    5,
                    960 / reduce_divider,
                    160 / reduce_divider,
                    true,
                    HS,
                    1,
                    dilation,
                ),
            ];
            let last_channel =
                adjust_channels(1280 / reduce_divider, width_mult);
            (settings, last_channel)
        }
        MobileNetV3Arch::Small => {
            let settings = vec![
                bneck_conf(16, 3, 16, 16, true, RE, 2, 1),
                bneck_conf(16, 3, 72, 24, false, RE, 2, 1),
                bneck_conf(24, 3, 88, 24, false, RE, 1, 1),
                bneck_conf(24, 5, 96, 40, true, HS, 2, 1),
                bneck_conf(40, 5, 240, 40, true, HS, 1, 1),
                bneck_conf(40, 5, 240, 40, true, HS, 1, 1),
                bneck_conf(40, 5, 120, 48, true, HS, 1, 1),
                bneck_conf(48, 5, 144, 48, true, HS, 1, 1),
                bneck_conf(
                    48,
                    5,
                    288,
                    96 / reduce_divider,
                    true,
                    HS,
                    2,
                    dilation,
                ),
                bneck_conf(
                    96 / reduce_divider,
                    5,
                    576 / reduce_divider,
                    96 / reduce_divider,
                    true,
                    HS,
                    1,
                    dilation,
                ),
                bneck_conf(
                    96 / reduce_divider,
                    5,
                    576 / reduce_divider,
                    96 / reduce_divider,
                    true,
                    HS,
                    1,
                    dilation,
                ),
            ];
            let last_channel =
                adjust_channels(1024 / reduce_divider, width_mult);
            (settings, last_channel)
        }
    }
}

#[cfg(feature = "pretrained")]
fn load_weights_record<B: Backend>(
    weights: &weights::Weights,
    device: &Device<B>,
) -> Result<MobileNetV3Record<B>, RecorderError> {
    let torch_weights = weights.download().map_err(|err| {
        RecorderError::Unknown(format!(
            "Could not download weights.\nError: {err}"
        ))
    })?;

    let load_args = LoadArgs::new(torch_weights)
        // Initial Conv and BatchNorm layers
        .with_key_remap("features\\.0\\.0\\.(.+)", "features.0.conv.$1")
        .with_key_remap("features\\.0\\.1\\.(.+)", "features.0.bn.$1")
        // Feature layer 1 does not have an expand conv (only layers 2-15)
        // So we only deal with the depthwise -> project convolutions
        .with_key_remap(
            "features\\.1\\.block\\.0\\.0\\.(.+)",
            "features.1.depthwise_conv.conv.$1",
        )
        .with_key_remap(
            "features\\.1\\.block\\.0\\.1\\.(.+)",
            "features.1.depthwise_conv.bn.$1",
        )
        .with_key_remap(
            "features\\.1\\.block\\.1\\.0\\.(.+)",
            "features.1.project_conv.conv.$1",
        )
        .with_key_remap(
            "features\\.1\\.block\\.1\\.1\\.(.+)",
            "features.1.project_conv.bn.$1",
        )
        // Feature layers 2-15 below
        // Inverted Residual Blocks - Expand Conv
        .with_key_remap(
            "features\\.([2-9]|1[0-5])\\.block\\.0\\.0\\.(.+)",
            "features.$1.expand_conv.conv.$2",
        )
        .with_key_remap(
            "features\\.([2-9]|1[0-5])\\.block\\.0\\.1\\.(.+)",
            "features.$1.expand_conv.bn.$2",
        )
        // Inverted Residual Blocks - Depthwise Conv
        .with_key_remap(
            "features\\.([2-9]|1[0-5])\\.block\\.1\\.0\\.(.+)",
            "features.$1.depthwise_conv.conv.$2",
        )
        .with_key_remap(
            "features\\.([2-9]|1[0-5])\\.block\\.1\\.1\\.(.+)",
            "features.$1.depthwise_conv.bn.$2",
        )
        // NOTE: block.2 remaps to project_conv when se_layer is not present, but block.3 otherwise
        // For large, se_layer is present in features.[4,5,6,11,12,13,14,15]
        // Squeeze and Excitation layers
        .with_key_remap(
            "features\\.(1[0-6]|[1-9])\\.block\\.2\\.fc1\\.(.+)",
            "features.$1.se_layer.fc1.$2",
        )
        .with_key_remap(
            "features\\.(1[0-6]|[1-9])\\.block\\.2\\.fc2\\.(.+)",
            "features.$1.se_layer.fc2.$2",
        )
        // When se_layer is not present, project_conv is in block.2
        // So for features.[1,2,3,7,8,9,10] it will be in block.2
        // Inverted Residual Blocks - Project Conv
        .with_key_remap(
            "features\\.([1-3]|[7-9]|10)\\.block\\.2\\.0\\.(.+)",
            "features.$1.project_conv.conv.$2",
        )
        .with_key_remap(
            "features\\.([1-3]|[7-9]|10)\\.block\\.2\\.1\\.(.+)",
            "features.$1.project_conv.bn.$2",
        )
        .with_key_remap(
            "features\\.([4-6]|1[1-5])\\.block\\.3\\.0\\.(.+)",
            "features.$1.project_conv.conv.$2",
        )
        .with_key_remap(
            "features\\.([4-6]|1[1-5])\\.block\\.3\\.1\\.(.+)",
            "features.$1.project_conv.bn.$2",
        )
        // Last Conv Layer
        .with_key_remap("features\\.16\\.0\\.(.+)", "features.16.conv.$1")
        .with_key_remap("features\\.16\\.1\\.(.+)", "features.16.bn.$1")
        // Classifier
        .with_key_remap("classifier\\.0\\.(.+)", "classifier.fc1.$1")
        .with_key_remap("classifier\\.3\\.(.+)", "classifier.fc2.$1");

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args, device)?;
    Ok(record)
}
#[derive(Config, Debug)]
pub struct MobileNetV3Config {
    #[config(default = "1000")]
    num_classes: usize,
}

impl MobileNetV3Config {
    fn create<B: Backend>(
        inverted_residual_setting: Vec<InvertedResidualConfig>,
        last_channel: usize,
        num_classes: usize,
        device: &Device<B>,
    ) -> MobileNetV3<B> {
        if inverted_residual_setting.is_empty() {
            panic!("The inverted_residual_setting can not not be empty");
        }

        let mut features: Vec<LayerType<B>> = Vec::new();

        // Building first layer
        let firstconv_output_channels =
            inverted_residual_setting[0].input_channels;
        features.push(LayerType::ConvBNActivation(
            ConvBNActivationConfig::new(
                3, // in_channels
                firstconv_output_channels,
            )
            .with_stride(2)
            .with_activation(Activation::Hardswish(Hardswish::new()))
            .init(device),
        ));

        // Building inverted residual blocks
        for cnf in inverted_residual_setting.iter() {
            features.push(LayerType::InvertedResidual(cnf.init(device)));
        }

        // Building last several layers
        let lastconv_input_channels =
            inverted_residual_setting.last().unwrap().out_channels;
        let lastconv_output_channels = 6 * lastconv_input_channels;
        features.push(LayerType::ConvBNActivation(
            ConvBNActivationConfig::new(
                lastconv_input_channels,
                lastconv_output_channels,
            )
            .with_kernel_size(1)
            .with_activation(Activation::Hardswish(Hardswish::new()))
            .init(device),
        ));

        let classifier = Classifier {
            fc1: LinearConfig::new(lastconv_output_channels, last_channel)
                .init(device),
            activation: Hardswish::new(),
            dropout: DropoutConfig::new(0.2).init(),
            fc2: LinearConfig::new(last_channel, num_classes).init(device),
        };

        return MobileNetV3 {
            features,
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            classifier,
        };
    }
    pub fn init_large<B: Backend>(&self, device: &B::Device) -> MobileNetV3<B> {
        let (inverted_residual_setting, last_channel) =
            mobilenet_v3_conf(MobileNetV3Arch::Large, 1.0, false, false);
        return Self::create(
            inverted_residual_setting,
            last_channel,
            self.num_classes,
            device,
        );
    }

    pub fn init_small<B: Backend>(&self, device: &Device<B>) -> MobileNetV3<B> {
        let (inverted_residual_setting, last_channel) =
            mobilenet_v3_conf(MobileNetV3Arch::Small, 1.0, false, false);
        return Self::create(
            inverted_residual_setting,
            last_channel,
            self.num_classes,
            device,
        );
    }
}

#[cfg(feature = "pretrained")]
#[derive(Config, Debug)]
pub struct MobileNetV3PretrainedConfig {
    weights_type: weights::MobileNetV3,
}

#[cfg(feature = "pretrained")]
impl MobileNetV3PretrainedConfig {
    pub fn init<B: Backend>(
        &self,
        device: &Device<B>,
    ) -> Result<MobileNetV3<B>, RecorderError> {
        let weights = self.weights_type.weights();
        let record = load_weights_record(&weights, device)?;

        let config =
            MobileNetV3Config::new().with_num_classes(weights.num_classes);

        let model = match self.weights_type {
            weights::MobileNetV3::PyTorchLarge => config.init_large(device),
            weights::MobileNetV3::PyTorchSmall => config.init_small(device),
        }
        .load_record(record);

        return Ok(model);
    }
}

#[derive(Module, Debug)]
pub struct MobileNetV3<B: Backend> {
    features: Vec<LayerType<B>>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Classifier<B>,
}

impl<B: Backend> MobileNetV3<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = input.clone();
        for layer in &self.features {
            x = layer.forward(x);
        }
        x = self.avgpool.forward(x);

        let batch_size = x.shape().dims[0];
        let num_elements = x.shape().num_elements() / batch_size;
        let reshaped = x.reshape([batch_size, num_elements]);

        return self.classifier.forward(reshaped);
    }
}
