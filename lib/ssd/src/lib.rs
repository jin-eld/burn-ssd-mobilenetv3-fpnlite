//! SSD / SSDLite implementation for Burn.
//!
//! This crate provides:
//! - SSDLite head (classification + regression)
//! - Anchor generator
//! - Box decoder
//! - Full SSDLite MobileNetV3 model

pub mod anchors;
pub mod decoder;
pub mod head;
pub mod model;
pub mod ops;

// Re‑exports for convenience
pub use anchors::AnchorGenerator;
pub use decoder::{BoxDecoder, DecodeConfig};
pub use head::SSDLiteHead;
pub use model::SSDLiteMobileNetV3;
