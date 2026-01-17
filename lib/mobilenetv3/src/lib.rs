pub mod imagenet;
mod model;
pub use model::{
    mobilenetv3::{MobileNetV3, MobileNetV3Arch, MobileNetV3Config},
    Relu6,
};

#[cfg(feature = "pretrained")]
pub use model::mobilenetv3::MobileNetV3PretrainedConfig;

#[cfg(feature = "pretrained")]
pub use model::weights;
