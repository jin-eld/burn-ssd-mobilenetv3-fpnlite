//! SSD / SSDLite implementation for Burn.
//!
//! This crate provides:
//! - SSDLite head (classification + regression)
//! - Anchor generator
//! - Box decoder
//! - Full SSDLite MobileNetV3 model

pub mod anchors;
pub mod decoder;
pub mod encoder;
pub mod head;
pub mod model;
pub mod ops;

// Re‑exports for convenience
pub use anchors::AnchorGenerator;
pub use decoder::{BoxDecoder, DecodeConfig};
pub use encoder::{BoxEncoder, EncodeConfig};
pub use head::SSDLiteHead;
pub use model::SSDLiteMobileNetV3;

#[cfg(test)]
mod tests {
    use crate::{
        decoder::{BoxDecoder, DecodeConfig},
        encoder::{BoxEncoder, EncodeConfig},
    };
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;

    type B = Wgpu;

    fn device() -> WgpuDevice {
        WgpuDevice::default()
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let device = device();

        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let gt =
            Tensor::<B, 3>::from_floats([[[0.6, 0.4, 0.25, 0.15]]], &device);

        let enc = BoxEncoder::new(EncodeConfig::default());
        let deltas = enc.encode::<B>(gt.clone(), anchors.clone());

        let dec = BoxDecoder::new(DecodeConfig::default());
        let out = dec.decode::<B>(deltas, anchors);

        let v = out.into_data().to_vec::<f32>().unwrap();
        let g = gt.into_data().to_vec::<f32>().unwrap();

        for i in 0..4 {
            assert!((v[i] - g[i]).abs() < 1e-5);
        }
    }
}
