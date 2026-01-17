use burn::{
    tensor::backend::Backend,
    tensor::{Int, Tensor},
};

#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// Variance for center coordinates (tx, ty).
    pub center_variance: f32,
    /// Variance for size coordinates (tw, th).
    pub size_variance: f32,
    /// Whether to clip decoded boxes to [0, 1].
    pub clip: bool,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        return Self {
            center_variance: 0.1,
            size_variance: 0.2,
            clip: true,
        };
    }
}

#[derive(Debug, Clone)]
pub struct BoxDecoder {
    cfg: DecodeConfig,
}

impl BoxDecoder {
    pub fn new(cfg: DecodeConfig) -> Self {
        return Self { cfg };
    }

    /// Decode box deltas relative to anchors.
    ///
    /// - `bbox_deltas`: [N, A, 4] (tx, ty, tw, th)
    /// - `anchors`:     [A, 4]    (cx, cy, w, h) in normalized coords
    /// - returns:       [N, A, 4] (cx, cy, w, h) in normalized coords
    pub fn decode<B: Backend>(
        &self,
        bbox_deltas: Tensor<B, 3>, // [N, A, 4]
        anchors: Tensor<B, 2>,     // [A, 4]
    ) -> Tensor<B, 3> {
        let center_var = self.cfg.center_variance;
        let size_var = self.cfg.size_variance;

        // anchors: [A,4] -> [1,A,4]
        let anchors: Tensor<B, 3> = anchors.unsqueeze_dim(0);

        let device = bbox_deltas.device();

        let idx0 = Tensor::<B, 1, Int>::from_data([0], &device);
        let idx1 = Tensor::<B, 1, Int>::from_data([1], &device);
        let idx2 = Tensor::<B, 1, Int>::from_data([2], &device);
        let idx3 = Tensor::<B, 1, Int>::from_data([3], &device);

        // deltas [N,A,4] -> [N,A,1]
        let tx = bbox_deltas.clone().select(2, idx0.clone());
        let ty = bbox_deltas.clone().select(2, idx1.clone());
        let tw = bbox_deltas.clone().select(2, idx2.clone());
        let th = bbox_deltas.clone().select(2, idx3.clone());

        // anchors [1,A,4] -> [1,A,1]
        let cx = anchors.clone().select(2, idx0.clone());
        let cy = anchors.clone().select(2, idx1.clone());
        let wa = anchors.clone().select(2, idx2.clone());
        let ha = anchors.clone().select(2, idx3.clone());

        // SSD decode
        let cx_dec = tx.mul_scalar(center_var).mul(wa.clone()).add(cx.clone());
        let cy_dec = ty.mul_scalar(center_var).mul(ha.clone()).add(cy.clone());
        let w_dec = tw.mul_scalar(size_var).exp().mul(wa);
        let h_dec = th.mul_scalar(size_var).exp().mul(ha);

        // Concatenate along last dimension → [N, A, 4]
        let mut decoded = Tensor::cat(vec![cx_dec, cy_dec, w_dec, h_dec], 2);

        if self.cfg.clip {
            decoded = decoded.clamp(0.0, 1.0);
        }

        return decoded;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::tensor::Tensor;

    type B = Wgpu;

    fn device() -> WgpuDevice {
        WgpuDevice::default()
    }

    #[test]
    fn test_decode_shapes() {
        let device = device();

        // bbox_deltas: [2, 3, 4]
        let deltas = Tensor::<B, 3>::from_floats(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ],
                [
                    [0.2, 0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8, 0.9],
                    [1.0, 1.1, 1.2, 1.3],
                ],
            ],
            &device,
        );

        // anchors: [3, 4]
        let anchors = Tensor::<B, 2>::from_floats(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.3, 0.1, 0.1],
                [0.7, 0.7, 0.3, 0.3],
            ],
            &device,
        );

        let decoder = BoxDecoder::new(DecodeConfig::default());
        let out = decoder.decode::<B>(deltas, anchors);

        assert_eq!(out.dims(), [2, 3, 4]);
    }

    #[test]
    fn test_decode_math_simple_case() {
        let device = device();

        // deltas: [1,1,4]
        let deltas =
            Tensor::<B, 3>::from_floats([[[0.0, 0.0, 0.0, 0.0]]], &device);

        // anchors: [1,4]
        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let decoder = BoxDecoder::new(DecodeConfig::default());
        let out = decoder.decode::<B>(deltas, anchors);

        // Extract values as Vec<f32>
        let values = out.into_data().to_vec::<f32>().unwrap();

        assert!((values[0] - 0.5).abs() < 1e-6); // cx
        assert!((values[1] - 0.5).abs() < 1e-6); // cy
        assert!((values[2] - 0.2).abs() < 1e-6); // w
        assert!((values[3] - 0.2).abs() < 1e-6); // h
    }

    #[test]
    fn test_decode_nonzero_deltas() {
        let device = device();

        // One batch, one anchor
        let deltas = Tensor::<B, 3>::from_floats(
            [[[0.2, -0.1, 0.5, -0.3]]], // tx, ty, tw, th
            &device,
        );

        let anchors = Tensor::<B, 2>::from_floats(
            [[0.4, 0.6, 0.2, 0.1]], // cx, cy, w, h
            &device,
        );

        let cfg = DecodeConfig::default();
        let decoder = BoxDecoder::new(cfg);

        let out = decoder.decode::<B>(deltas, anchors);
        let v = out.into_data().to_vec::<f32>().unwrap();

        // Expected math:
        // cx' = 0.2 * 0.1 * 0.2 + 0.4 = 0.404
        // cy' = -0.1 * 0.1 * 0.1 + 0.6 = 0.599
        // w'  = exp(0.5 * 0.2) * 0.2 = exp(0.1) * 0.2
        // h'  = exp(-0.3 * 0.2) * 0.1 = exp(-0.06) * 0.1

        let expected_cx = 0.404;
        let expected_cy = 0.599;
        let expected_w = (0.1f32).exp() * 0.2;
        let expected_h = (-0.06f32).exp() * 0.1;

        assert!((v[0] - expected_cx).abs() < 1e-6);
        assert!((v[1] - expected_cy).abs() < 1e-6);
        assert!((v[2] - expected_w).abs() < 1e-6);
        assert!((v[3] - expected_h).abs() < 1e-6);
    }

    #[test]
    fn test_decode_clipping() {
        let device = device();

        // tx = 50 pushes cx far beyond 1.0
        let deltas =
            Tensor::<B, 3>::from_floats([[[50.0, 50.0, 0.0, 0.0]]], &device);

        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let cfg = DecodeConfig {
            clip: true,
            ..Default::default()
        };
        let decoder = BoxDecoder::new(cfg);

        let out = decoder.decode::<B>(deltas, anchors);
        let v = out.into_data().to_vec::<f32>().unwrap();

        assert!((v[0] - 1.0).abs() < 1e-6); // cx clipped
        assert!((v[1] - 1.0).abs() < 1e-6); // cy clipped
    }

    #[test]
    fn test_decode_multi_batch_multi_anchor() {
        let device = device();

        // 2 batches, 3 anchors
        let deltas = Tensor::<B, 3>::from_floats(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0],
                ],
                [
                    [0.4, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0, 0.0],
                ],
            ],
            &device,
        );

        let anchors = Tensor::<B, 2>::from_floats(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
            ],
            &device,
        );

        let decoder = BoxDecoder::new(DecodeConfig::default());
        let out = decoder.decode::<B>(deltas, anchors);

        assert_eq!(out.dims(), [2, 3, 4]);

        let v = out.into_data().to_vec::<f32>().unwrap();

        // Spot‑check a few values:
        // cx' = tx * 0.1 * w + cx
        // For batch 0, anchor 0: tx=0.1, w=0.1, cx=0.1 → cx'=0.101
        assert!((v[0] - 0.101).abs() < 1e-6);
    }

    #[test]
    fn test_decode_clipping_negative() {
        let device = device();

        // tx = -50 pushes cx far below 0
        let deltas =
            Tensor::<B, 3>::from_floats([[[-50.0, -50.0, 0.0, 0.0]]], &device);

        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let cfg = DecodeConfig {
            clip: true,
            ..Default::default()
        };
        let decoder = BoxDecoder::new(cfg);

        let out = decoder.decode::<B>(deltas, anchors);
        let v = out.into_data().to_vec::<f32>().unwrap();

        assert!((v[0] - 0.0).abs() < 1e-6); // cx clipped
        assert!((v[1] - 0.0).abs() < 1e-6); // cy clipped
    }

    #[test]
    fn test_decode_clipping_mixed() {
        let device = device();

        // tx pushes cx > 1, ty keeps cy inside [0,1]
        let deltas =
            Tensor::<B, 3>::from_floats([[[50.0, 0.0, 0.0, 0.0]]], &device);

        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let cfg = DecodeConfig {
            clip: true,
            ..Default::default()
        };
        let decoder = BoxDecoder::new(cfg);

        let out = decoder.decode::<B>(deltas, anchors);
        let v = out.into_data().to_vec::<f32>().unwrap();

        assert!((v[0] - 1.0).abs() < 1e-6); // cx clipped
        assert!((v[1] - 0.5).abs() < 1e-6); // cy unchanged
    }

    #[test]
    fn test_decode_clipping_sizes() {
        let device = device();

        // tw/th large enough to blow up w/h
        let deltas =
            Tensor::<B, 3>::from_floats([[[0.0, 0.0, 20.0, 20.0]]], &device);

        let anchors =
            Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.2, 0.2]], &device);

        let cfg = DecodeConfig {
            clip: true,
            ..Default::default()
        };
        let decoder = BoxDecoder::new(cfg);

        let out = decoder.decode::<B>(deltas, anchors);
        let v = out.into_data().to_vec::<f32>().unwrap();

        assert!((v[2] - 1.0).abs() < 1e-6); // w clipped
        assert!((v[3] - 1.0).abs() < 1e-6); // h clipped
    }
}
