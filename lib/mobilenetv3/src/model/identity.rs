use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct IdentityConfig {}

impl IdentityConfig {
    pub fn init(&self) -> Identity {
        return Identity {};
    }
}

#[derive(Module, Clone, Debug)]
pub struct Identity {}

impl Identity {
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        return input;
    }
}
