//! FPNLite feature pyramid for MobileNetV3-based SSD detectors.

pub mod block;
mod fpnlite;

pub use block::DepthwiseSeparableBlock;
pub use fpnlite::{FpnLite, FpnLiteConfig};
