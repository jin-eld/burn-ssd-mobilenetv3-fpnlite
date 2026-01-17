/// Convert [cx, cy, w, h] → [x1, y1, x2, y2].
pub fn cxcywh_to_xyxy(boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
    return boxes
        .iter()
        .map(|b| {
            let cx = b[0];
            let cy = b[1];
            let w = b[2];
            let h = b[3];

            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            [x1, y1, x2, y2]
        })
        .collect();
}

/// IoU between two boxes in [x1, y1, x2, y2].
pub fn iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;

    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);

    if inter <= 0.0 {
        return 0.0;
    } else {
        return inter / (area_a + area_b - inter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cxcywh_to_xyxy_centered() {
        let boxes = vec![[0.5, 0.5, 0.4, 0.2]];
        let xyxy = cxcywh_to_xyxy(&boxes);

        let b = xyxy[0];
        assert!((b[0] - 0.3).abs() < 1e-6);
        assert!((b[1] - 0.4).abs() < 1e-6);
        assert!((b[2] - 0.7).abs() < 1e-6);
        assert!((b[3] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_iou_identical_boxes() {
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [0.0, 0.0, 1.0, 1.0];
        assert!((iou_xyxy(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = [0.0, 0.0, 0.2, 0.2];
        let b = [0.8, 0.8, 1.0, 1.0];
        assert!(iou_xyxy(&a, &b) == 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = [0.0, 0.0, 0.5, 0.5];
        let b = [0.25, 0.25, 0.75, 0.75];

        // Intersection = 0.25 * 0.25 = 0.0625
        // Area A = 0.25, Area B = 0.25
        // IoU = 0.0625 / (0.25 + 0.25 - 0.0625) = 0.0625 / 0.4375 = 0.142857
        let expected = 0.142857;
        assert!((iou_xyxy(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_iou_symmetry() {
        let a = [0.1, 0.1, 0.4, 0.4];
        let b = [0.2, 0.2, 0.5, 0.5];
        assert!((iou_xyxy(&a, &b) - iou_xyxy(&b, &a)).abs() < 1e-6);
    }
}
