//! Input augmentation for SSD training.
//!
//! Pixel-level transforms and their bounding-box counterparts live here so
//! that images and boxes can never drift out of sync. Photometric distortions
//! (brightness / contrast / hue) do not move pixels spatially and therefore
//! require no box adjustment; geometric transforms (horizontal flip) must be
//! mirrored on the normalized cxcywh boxes.

use image::RgbImage;
use rand::RngExt;

/// One sample's augmentation draw.
#[derive(Clone, Debug, PartialEq)]
pub struct AugmentationParams {
    /// Whether to mirror the image horizontally.
    pub hflip: bool,
    /// Additive brightness offset in [-32, 32].
    pub brightness: i32,
    /// Multiplicative contrast factor in [0.5, 1.5].
    pub contrast: f32,
    /// Hue rotation in degrees in [-18, 18].
    pub hue: i32,
}

impl AugmentationParams {
    /// Draw a random augmentation (SSD-style ranges).
    pub fn random<R: RngExt>(rng: &mut R) -> Self {
        return Self {
            hflip: rng.random_bool(0.5),
            brightness: rng.random_range(-32..32),
            contrast: rng.random_range(0.5..1.5),
            hue: rng.random_range(-18..18),
        };
    }

    /// No-op parameters: identical pixels, identical boxes.
    pub fn identity() -> Self {
        return Self {
            hflip: false,
            brightness: 0,
            contrast: 1.0,
            hue: 0,
        };
    }
}

/// Apply photometric distortion. Pixel-only: safe for bounding boxes.
/// Neutral parameters are skipped so that identity() is exactly identity.
pub fn distort_image(img: RgbImage, p: &AugmentationParams) -> RgbImage {
    let mut img = img;
    if p.brightness != 0 {
        img = image::imageops::brighten(&img, p.brightness);
    }
    if (p.contrast - 1.0).abs() > f32::EPSILON {
        img = image::imageops::contrast(&img, p.contrast);
    }
    if p.hue != 0 {
        img = image::imageops::huerotate(&img, p.hue);
    }
    return img;
}

/// Mirror an RGB image horizontally.
pub fn flip_image(img: &RgbImage) -> RgbImage {
    return image::imageops::flip_horizontal(img);
}

/// Mirror normalized cxcywh boxes horizontally: cx -> 1 - cx.
/// cy, w and h are unchanged by a horizontal flip.
pub fn flip_boxes(boxes: &mut [[f32; 4]]) {
    for b in boxes.iter_mut() {
        b[0] = 1.0 - b[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn left_white_right_black() -> RgbImage {
        let mut img = RgbImage::new(8, 8);
        for x in 0..8u32 {
            for y in 0..8u32 {
                let v = if x < 4 { 255 } else { 0 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        return img;
    }

    #[test]
    fn test_flip_boxes_flips_cx_only() {
        let mut boxes = vec![[0.25, 0.4, 0.2, 0.3]];
        flip_boxes(&mut boxes);
        assert_eq!(boxes[0], [0.75, 0.4, 0.2, 0.3]);
    }

    #[test]
    fn test_flip_boxes_is_involution() {
        let original = vec![[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.05, 0.05]];
        let mut boxes = original.clone();
        flip_boxes(&mut boxes);
        flip_boxes(&mut boxes);
        // f32 arithmetic on repeating decimals (like 0.1) introduces tiny
        // rounding errors (e.g. 1.0 - 0.9 = 0.100000024). Compare with epsilon.
        for (b, o) in boxes.iter().zip(original.iter()) {
            for i in 0..4 {
                assert!(
                    (b[i] - o[i]).abs() < 1e-6,
                    "expected {}, got {}",
                    o[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn test_flip_boxes_empty_is_noop() {
        let mut boxes: Vec<[f32; 4]> = vec![];
        flip_boxes(&mut boxes);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_flipped_image_and_boxes_stay_in_sync() {
        let img = left_white_right_black();
        // box covering exactly the white (left) half: x in [0, 4)
        let mut boxes = vec![[0.25, 0.5, 0.5, 1.0]];
        let flipped = flip_image(&img);
        flip_boxes(&mut boxes);
        // after flip the white half is on the right: x in [4, 8)
        for x in 0..8u32 {
            for y in 0..8u32 {
                let expected = if x >= 4 { 255 } else { 0 };
                assert_eq!(
                    flipped.get_pixel(x, y).0,
                    [expected, expected, expected]
                );
            }
        }
        // flipped box must cover exactly the white region
        let cx = boxes[0][0];
        let w = boxes[0][2];
        let x0 = (cx - w / 2.0) * 8.0;
        let x1 = (cx + w / 2.0) * 8.0;
        assert_eq!(x0, 4.0);
        assert_eq!(x1, 8.0);
        for x in (x0 as u32)..(x1 as u32) {
            assert_eq!(flipped.get_pixel(x, 0).0, [255, 255, 255]);
        }
    }

    #[test]
    fn test_identity_params_leave_pixels_unchanged() {
        let img = left_white_right_black();
        let expected = img.clone();
        let out = distort_image(img, &AugmentationParams::identity());
        assert_eq!(out.into_raw(), expected.into_raw());
    }

    #[test]
    fn test_brightness_is_applied() {
        let img = RgbImage::from_pixel(4, 4, Rgb([100, 100, 100]));
        let p = AugmentationParams {
            hflip: false,
            brightness: 50,
            contrast: 1.0,
            hue: 0,
        };
        let out = distort_image(img, &p);
        assert_eq!(out.get_pixel(0, 0).0, [150, 150, 150]);
    }

    #[test]
    fn test_random_params_stay_in_ranges() {
        let mut rng = rand::rng();
        for _ in 0..64 {
            let p = AugmentationParams::random(&mut rng);
            assert!((-32..32).contains(&p.brightness));
            assert!((0.5..1.5).contains(&p.contrast));
            assert!((-18..18).contains(&p.hue));
        }
    }
}
