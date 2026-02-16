pub mod dataset;
pub mod loss;
pub mod target;
pub mod training;

#[cfg(test)]
mod tests {
    use crate::target::encoder::{BoxEncoder, EncodeConfig};
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;
    use ssd::decoder::{BoxDecoder, DecodeConfig};

    type B = Wgpu;

    #[test]
    fn test_encode_decode_roundtrip() {
        let device = WgpuDevice::default();

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
