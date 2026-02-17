pub mod dataset;
pub mod loss;
pub mod target;
pub mod training;

#[cfg(test)]
mod tests {
    use crate::target::encoder::{BoxEncoder, EncodeConfig};
    use burn::tensor::{Device, Tensor};
    use ssd::decoder::{BoxDecoder, DecodeConfig};

    #[test]
    fn test_encode_decode_roundtrip() {
        let device = Device::default();

        let anchors = Tensor::<2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let gt = Tensor::<3>::from_floats([[[0.6, 0.4, 0.25, 0.15]]], &device);

        let enc = BoxEncoder::new(EncodeConfig::default());
        let deltas = enc.encode(gt.clone(), anchors.clone());

        let dec = BoxDecoder::new(DecodeConfig::default());
        let out = dec.decode(deltas, anchors);

        let v = out.into_data().to_vec::<f32>().unwrap();
        let g = gt.into_data().to_vec::<f32>().unwrap();

        for i in 0..4 {
            assert!((v[i] - g[i]).abs() < 1e-5);
        }
    }
}
