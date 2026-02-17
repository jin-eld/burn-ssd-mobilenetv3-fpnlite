use burn::{config::Config, module::Module, tensor::Tensor};

#[derive(Config, Debug)]
pub struct IdentityConfig {}

impl IdentityConfig {
    pub fn init(&self) -> Identity {
        return Identity {};
    }
}

#[derive(Module, Debug)]
pub struct Identity {}

impl Identity {
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        return input;
    }
}
