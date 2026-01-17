#[derive(Debug, Clone)]
pub struct AnchorGenerator {
    /// Minimum scale (0.2 in TF SSD)
    pub min_scale: f32,
    /// Maximum scale (0.95 in TF SSD)
    pub max_scale: f32,
    /// Global aspect ratios (TF allows per-layer, but this is enough for SSD-Lite)
    pub aspect_ratios: Vec<f32>,
    /// Whether to clip anchors to [0,1]
    pub clip: bool,
    /// TF SSD: whether to use special small boxes on the lowest layer
    pub reduce_boxes_in_lowest_layer: bool,
    /// TF SSD: extra anchor with interpolated scale and this aspect ratio (usually 1.0)
    pub interpolated_scale_aspect_ratio: Option<f32>,
}

impl AnchorGenerator {
    pub fn new(
        min_scale: f32,
        max_scale: f32,
        aspect_ratios: Vec<f32>,
        clip: bool,
        reduce_boxes_in_lowest_layer: bool,
        interpolated_scale_aspect_ratio: Option<f32>,
    ) -> Self {
        return Self {
            min_scale,
            max_scale,
            aspect_ratios,
            clip,
            reduce_boxes_in_lowest_layer,
            interpolated_scale_aspect_ratio,
        };
    }

    /// Compute SSD scales for each feature level.
    fn compute_scales(&self, num_levels: usize) -> Vec<f32> {
        let m = num_levels as f32;
        let mut scales = Vec::new();

        for k in 0..num_levels {
            let sk = self.min_scale
                + (self.max_scale - self.min_scale) * (k as f32) / (m - 1.0);
            scales.push(sk);
        }

        return scales;
    }

    /// Extra SSD scale for aspect ratio 1.0
    fn extra_scale(s_k: f32, s_k1: f32) -> f32 {
        return (s_k * s_k1).sqrt();
    }

    /// Generate anchors in normalized [cx, cy, w, h] format.
    ///
    /// IMPORTANT:
    /// - `feature_map_sizes` must come from the actual FPN outputs.
    /// - This ensures anchors always match SSD head predictions.
    pub fn generate(
        &self,
        feature_map_sizes: &[(usize, usize)],
    ) -> Vec<[f32; 4]> {
        let mut anchors = Vec::new();
        let num_levels = feature_map_sizes.len();
        let scales = self.compute_scales(num_levels);

        for (k, &(fh, fw)) in feature_map_sizes.iter().enumerate() {
            let sk = scales[k];
            let sk1 = if k + 1 < scales.len() {
                scales[k + 1]
            } else {
                1.0 // TF SSD convention for last layer
            };

            let s_extra = Self::extra_scale(sk, sk1);

            for y in 0..fh {
                for x in 0..fw {
                    let cx = (x as f32 + 0.5) / fw as f32;
                    let cy = (y as f32 + 0.5) / fh as f32;

                    // Lowest layer special-case (TF's reduce_boxes_in_lowest_layer)
                    if self.reduce_boxes_in_lowest_layer && k == 0 {
                        let small_scales = [
                            0.1_f32,
                            self.min_scale,
                            Self::extra_scale(0.1, self.min_scale),
                        ];
                        for &s in &small_scales {
                            anchors.push([cx, cy, s, s]);
                        }
                    } else {
                        // Standard aspect ratios
                        for &ar in &self.aspect_ratios {
                            let ar_sqrt = ar.sqrt();
                            let w = sk * ar_sqrt;
                            let h = sk / ar_sqrt;
                            anchors.push([cx, cy, w, h]);
                        }

                        // Extra SSD anchor with interpolated scale (usually AR=1.0)
                        if let Some(ar_extra) =
                            self.interpolated_scale_aspect_ratio
                        {
                            let ar_sqrt = ar_extra.sqrt();
                            let w = s_extra * ar_sqrt;
                            let h = s_extra / ar_sqrt;
                            anchors.push([cx, cy, w, h]);
                        }
                    }
                }
            }
        }

        if self.clip {
            for a in anchors.iter_mut() {
                a[0] = a[0].clamp(0.0, 1.0);
                a[1] = a[1].clamp(0.0, 1.0);
                a[2] = a[2].clamp(0.0, 1.0);
                a[3] = a[3].clamp(0.0, 1.0);
            }
        }

        return anchors;
    }
}
