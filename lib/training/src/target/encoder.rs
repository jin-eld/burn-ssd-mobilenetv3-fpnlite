use burn::tensor::Tensor;

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
    pub fn encode(
        &self,
        gt_boxes: Tensor<3>, // [N, A, 4]
        anchors: Tensor<2>,  // [A, 4]
    ) -> Tensor<3> {
        let center_var = self.cfg.center_variance;
        let size_var = self.cfg.size_variance;

        // anchors: [A,4] -> [1,A,4]
        let anchors = anchors.unsqueeze_dim(0);

        // GT components
        let gx = gt_boxes.clone().narrow(2, 0, 1);
        let gy = gt_boxes.clone().narrow(2, 1, 1);
        let gw = gt_boxes.clone().narrow(2, 2, 1);
        let gh = gt_boxes.clone().narrow(2, 3, 1);

        // Anchor components
        let ax = anchors.clone().narrow(2, 0, 1);
        let ay = anchors.clone().narrow(2, 1, 1);
        let aw = anchors.clone().narrow(2, 2, 1);
        let ah = anchors.clone().narrow(2, 3, 1);

        // SSD encode
        let tx = gx.sub(ax.clone()).div(aw.clone()).div_scalar(center_var);
        let ty = gy.sub(ay.clone()).div(ah.clone()).div_scalar(center_var);
        let tw = gw.div(aw.clone()).log().div_scalar(size_var);
        let th = gh.div(ah.clone()).log().div_scalar(size_var);

        Tensor::cat(vec![tx, ty, tw, th], 2)
    }
}
