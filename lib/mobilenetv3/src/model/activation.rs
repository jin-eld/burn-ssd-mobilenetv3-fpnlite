use super::identity::{Identity, IdentityConfig};
use burn::{
    module::Module,
    nn::Relu,
    tensor::{backend::Backend, Tensor},
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::fmt;

#[derive(Module, Debug, Clone, Default)]
pub struct Relu6 {
    relu: Relu,
}

impl Relu6 {
    pub fn new() -> Self {
        return Self { relu: Relu::new() };
    }

    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let x = self.relu.forward(input);
        return x.clamp_max(6.0);
    }
}

#[derive(Module, Debug, Clone, Default)]
pub struct Hardswish {
    relu6: Relu6,
}

impl Hardswish {
    pub fn new() -> Self {
        return Self {
            relu6: Relu6::new(),
        };
    }

    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // Hardswish: x * ReLU6(x + 3) / 6
        let x = self.relu6.forward(input.clone().add_scalar(3.0));
        return input.mul(x).div_scalar(6.0);
    }
}

#[derive(Module, Clone, Debug)]
pub enum Activation {
    Relu(Relu),
    Identity(Identity),
    Relu6(Relu6),
    Hardswish(Hardswish),
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self {
            Activation::Relu(layer) => layer.forward(input),
            Activation::Identity(layer) => layer.forward(input),
            Activation::Relu6(layer) => layer.forward(input),
            Activation::Hardswish(layer) => layer.forward(input),
        }
    }
}

impl Serialize for Activation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Activation::Relu(_) => serializer.serialize_str("Relu"),
            Activation::Identity(_) => serializer.serialize_str("Identity"),
            Activation::Relu6(_) => serializer.serialize_str("Relu6"),
            Activation::Hardswish(_) => serializer.serialize_str("Hardswish"),
        }
    }
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ActivationVisitor;

        impl<'de> Visitor<'de> for ActivationVisitor {
            type Value = Activation;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a valid activation function")
            }

            fn visit_str<E>(self, value: &str) -> Result<Activation, E>
            where
                E: de::Error,
            {
                match value {
                    "Relu" => Ok(Activation::Relu(Relu::new())),
                    "Identity" => {
                        Ok(Activation::Identity(IdentityConfig::new().init()))
                    }
                    "Relu6" => Ok(Activation::Relu6(Relu6::new())),
                    "Hardswish" => Ok(Activation::Hardswish(Hardswish::new())),
                    _ => Err(E::custom(format!(
                        "unknown activation function: {}",
                        value
                    ))),
                }
            }
        }

        return deserializer.deserialize_str(ActivationVisitor);
    }
}
