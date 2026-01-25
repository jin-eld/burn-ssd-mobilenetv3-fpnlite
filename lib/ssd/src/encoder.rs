use burn::{
    tensor::backend::Backend,
    tensor::{Int, Tensor},
};

#[derive(Debug, Clone)]
pub struct EncodeConfig {
    pub center_variance: f32,
    pub size_variance: f32,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self {
            center_variance: 0.1,
            size_variance: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoxEncoder {
    cfg: EncodeConfig,
}

impl BoxEncoder {
    pub fn new(cfg: EncodeConfig) -> Self {
        Self { cfg }
    }

    /// Encode GT boxes relative to anchors.
    ///
    /// - `gt_boxes`: [N, A, 4] (cx, cy, w, h)
    /// - `anchors`:  [A, 4]    (cx, cy, w, h)
    /// - returns:    [N, A, 4] (tx, ty, tw, th)
    pub fn encode<B: Backend>(
        &self,
        gt_boxes: Tensor<B, 3>, // [N, A, 4]
        anchors: Tensor<B, 2>,  // [A, 4]
    ) -> Tensor<B, 3> {
        let center_var = self.cfg.center_variance;
        let size_var = self.cfg.size_variance;

        // anchors: [A,4] -> [1,A,4]
        let anchors = anchors.unsqueeze_dim(0);

        let device = gt_boxes.device();

        let idx0 = Tensor::<B, 1, Int>::from_data([0], &device);
        let idx1 = Tensor::<B, 1, Int>::from_data([1], &device);
        let idx2 = Tensor::<B, 1, Int>::from_data([2], &device);
        let idx3 = Tensor::<B, 1, Int>::from_data([3], &device);

        // GT components
        let gx = gt_boxes.clone().select(2, idx0.clone());
        let gy = gt_boxes.clone().select(2, idx1.clone());
        let gw = gt_boxes.clone().select(2, idx2.clone());
        let gh = gt_boxes.clone().select(2, idx3.clone());

        // Anchor components
        let ax = anchors.clone().select(2, idx0.clone());
        let ay = anchors.clone().select(2, idx1.clone());
        let aw = anchors.clone().select(2, idx2.clone());
        let ah = anchors.clone().select(2, idx3.clone());

        // SSD encode
        let tx = gx.sub(ax.clone()).div(aw.clone()).div_scalar(center_var);
        let ty = gy.sub(ay.clone()).div(ah.clone()).div_scalar(center_var);
        let tw = gw.div(aw.clone()).log().div_scalar(size_var);
        let th = gh.div(ah.clone()).log().div_scalar(size_var);

        Tensor::cat(vec![tx, ty, tw, th], 2)
    }
}
