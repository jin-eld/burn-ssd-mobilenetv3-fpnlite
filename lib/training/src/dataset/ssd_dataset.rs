use burn::data::dataset::vision::{
    Annotation, BoundingBox, ImageDatasetItem, ImageFolderDataset,
};
use burn::data::dataset::{Dataset, DatasetError};
use burn::tensor::{Shape, TensorData};
use transforms::{coco_to_cxcywh_normalized, img_resize_ssd, scale_coco_boxes};

#[derive(Clone, Debug)]
pub struct SSDSample {
    pub image_data: TensorData, // [H, W, C] U8
    pub boxes: Vec<[f32; 4]>,   // normalized cxcywh
    pub labels: Vec<usize>,     // class ids
}

pub struct SSDDataset {
    inner: ImageFolderDataset,
    input_w: u32,
    input_h: u32,
}

impl SSDDataset {
    pub fn new(inner: ImageFolderDataset, input_w: u32, input_h: u32) -> Self {
        return Self {
            inner,
            input_w,
            input_h,
        };
    }
}

impl Dataset<SSDSample> for SSDDataset {
    fn len(&self) -> usize {
        return self.inner.len();
    }

    fn get(&self, index: usize) -> Result<SSDSample, DatasetError> {
        let item: ImageDatasetItem = self.inner.get(index)?;

        let pixels_u8: Vec<u8> = match item.image {
            burn::data::dataset::vision::PixelData::U8(bytes) => bytes,
            burn::data::dataset::vision::PixelData::U16(words) => {
                words.iter().map(|&w| (w >> 8) as u8).collect()
            }
            burn::data::dataset::vision::PixelData::F32(floats) => floats
                .iter()
                .map(|&f| (f.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        };

        let w = item.image_width as u32;
        let h = item.image_height as u32;
        let num_pixels = (w * h) as usize;

        // build DynamicImage, handling Grayscale (1-channel),
        // RGB (3-channel), and RGBA (4-channel)
        let dyn_img = match pixels_u8.len() {
            len if len == num_pixels => {
                let gray_img = image::GrayImage::from_raw(w, h, pixels_u8)
                    .expect("invalid grayscale image buffer size");
                image::DynamicImage::ImageLuma8(gray_img).to_rgb8().into()
            }
            len if len == num_pixels * 3 => {
                let rgb_img = image::RgbImage::from_raw(w, h, pixels_u8)
                    .expect("invalid RGB image buffer size");
                image::DynamicImage::ImageRgb8(rgb_img)
            }
            len if len == num_pixels * 4 => {
                let rgba_img = image::RgbaImage::from_raw(w, h, pixels_u8)
                    .expect("invalid RGBA image buffer size");
                image::DynamicImage::ImageRgba8(rgba_img).to_rgb8().into()
            }
            _ => panic!(
                "Unexpected pixel data length {} for {}x{} image (expected {}, {}, or {})",
                pixels_u8.len(),
                w,
                h,
                num_pixels,
                num_pixels * 3,
                num_pixels * 4
            ),
        };

        // extract boxes + labels from annotation
        let (mut boxes, labels): (Vec<[f32; 4]>, Vec<usize>) = match item
            .annotation
        {
            Annotation::BoundingBoxes(v) => {
                let b = v.iter().map(|bb: &BoundingBox| bb.coords).collect();
                let l = v.iter().map(|bb| bb.label).collect();
                (b, l)
            }
            _ => (Vec::new(), Vec::new()),
        };

        // resize image to SSD input size
        let resized = img_resize_ssd(&dyn_img, self.input_w, self.input_h);

        // scale boxes to new size (still in [x,y,w,h] pixels)
        scale_coco_boxes(
            &mut boxes,
            w as f32,
            h as f32,
            self.input_w as f32,
            self.input_h as f32,
        );

        // convert to normalized cxcywh
        let boxes = coco_to_cxcywh_normalized(
            &boxes,
            self.input_w as f32,
            self.input_h as f32,
        );

        // extract raw pixels as Vec<u8> for device-agnostic batching
        let rgb_img = resized.to_rgb8();
        let raw_pixels = rgb_img.into_raw();
        let image_data = TensorData::new(
            raw_pixels,
            Shape::new([self.input_h as usize, self.input_w as usize, 3]),
        );

        return Ok(SSDSample {
            image_data,
            boxes,
            labels,
        });
    }
}

impl SSDDataset {
    pub fn num_classes(&self) -> usize {
        // scan all annotations and find the maximum label
        let mut max_label = 0;

        for i in 0..self.inner.len() {
            if let Ok(item) = self.inner.get(i) {
                if let Annotation::BoundingBoxes(v) = item.annotation {
                    for bb in v {
                        max_label = max_label.max(bb.label);
                    }
                }
            }
        }

        // labels are zero-based, so +1 gives the class count
        return max_label + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataset::vision::ImageFolderDataset;

    const COCO_JSON: &str = "tests/dataset_coco.json";
    const COCO_IMAGES: &str = "tests/image_folder_coco";

    #[test]
    fn test_box_scaling_32_to_320() {
        let mut boxes = vec![[3.0, 4.0, 10.0, 12.0]];

        scale_coco_boxes(&mut boxes, 32.0, 32.0, 320.0, 320.0);

        let scaled = boxes[0];
        assert_eq!(scaled, [30.0, 40.0, 100.0, 120.0]);

        let norm = coco_to_cxcywh_normalized(&boxes, 320.0, 320.0)[0];

        assert!((norm[0] - 0.25).abs() < 1e-6);
        assert!((norm[1] - 0.3125).abs() < 1e-6);
        assert!((norm[2] - 0.3125).abs() < 1e-6);
        assert!((norm[3] - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_empty_annotations() {
        let coco =
            ImageFolderDataset::new_coco_detection(COCO_JSON, COCO_IMAGES)
                .unwrap();

        let dataset = SSDDataset::new(coco, 320, 320);

        // Find an image with no boxes
        let sample_result =
            (0..dataset.len())
                .map(|i| dataset.get(i))
                .find(|res| match res {
                    Ok(s) => s.boxes.is_empty(),
                    Err(_) => true, // stop here to catch and surface the error
                });

        let sample = match sample_result {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                panic!("Dataset error encountered during scan: {:?}", e)
            }
            None => panic!("expected at least one empty annotation"),
        };

        assert!(sample.boxes.is_empty());
        assert!(sample.labels.is_empty());
    }

    #[test]
    fn test_ssd_dataset_with_coco() {
        // load the same test dataset that Burn uses
        let coco =
            ImageFolderDataset::new_coco_detection(COCO_JSON, COCO_IMAGES)
                .unwrap();

        let dataset = SSDDataset::new(coco, 320, 320);

        let sample = dataset.get(0).expect("sample should exist");

        // image data shape is [H, W, C] before the batcher permutes it
        assert_eq!(sample.image_data.shape.dims(), [320, 320, 3]);

        // boxes normalized
        for b in &sample.boxes {
            assert!((0.0..=1.0).contains(&b[0]));
            assert!((0.0..=1.0).contains(&b[1]));
            assert!((0.0..=1.0).contains(&b[2]));
            assert!((0.0..=1.0).contains(&b[3]));
        }

        // labels exist
        assert!(!sample.labels.is_empty());
    }
}
