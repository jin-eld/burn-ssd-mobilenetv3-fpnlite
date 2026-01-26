use burn::data::dataset::vision::{
    Annotation, BoundingBox, ImageDatasetItem, ImageFolderDataset,
};
use burn::data::dataset::Dataset;
use burn::tensor::{backend::Backend, Tensor};
use transforms::{
    coco_to_cxcywh_normalized, img_resize_ssd, img_to_tensor, scale_coco_boxes,
};

#[derive(Clone, Debug)]
pub struct SSDSample<B: Backend> {
    pub image: Tensor<B, 3>,  // [C, H, W]
    pub boxes: Vec<[f32; 4]>, // normalized cxcywh
    pub labels: Vec<usize>,   // class ids
}

pub struct SSDDataset<B: Backend> {
    inner: ImageFolderDataset,
    input_w: u32,
    input_h: u32,
    device: B::Device,
}

impl<B: Backend> SSDDataset<B> {
    pub fn new(
        inner: ImageFolderDataset,
        input_w: u32,
        input_h: u32,
        device: B::Device,
    ) -> Self {
        return Self {
            inner,
            input_w,
            input_h,
            device,
        };
    }
}

impl<B: Backend> Dataset<SSDSample<B>> for SSDDataset<B> {
    fn len(&self) -> usize {
        return self.inner.len();
    }

    fn get(&self, index: usize) -> Option<SSDSample<B>> {
        let item: ImageDatasetItem = self.inner.get(index)?;

        // convert Vec<PixelDepth> -> Vec<u8>
        let pixels_u8: Vec<u8> = item
            .image
            .iter()
            .map(|p| u8::try_from(*p).expect("expected U8 pixel depth"))
            .collect();

        let w = item.image_width as u32;
        let h = item.image_height as u32;

        // build DynamicImage
        let rgb_img = image::RgbImage::from_raw(w, h, pixels_u8)
            .expect("invalid image buffer size");
        let dyn_img = image::DynamicImage::ImageRgb8(rgb_img);

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

        // convert resized image to tensor
        let image = img_to_tensor::<B>(resized, &self.device);

        return Some(SSDSample {
            image,
            boxes,
            labels,
        });
    }
}

impl<B: Backend> SSDDataset<B> {
    pub fn num_classes(&self) -> usize {
        // scan all annotations and find the maximum label
        let mut max_label = 0;

        for i in 0..self.inner.len() {
            if let Some(item) = self.inner.get(i) {
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
    use burn::backend::wgpu::Wgpu;
    use burn::data::dataset::vision::ImageFolderDataset;

    type B = Wgpu;

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

        let dataset = SSDDataset::<B>::new(coco, 320, 320, Default::default());

        // Find an image with no boxes
        let sample = (0..dataset.len())
            .find_map(|i| {
                let s = dataset.get(i)?;
                if s.boxes.is_empty() {
                    Some(s)
                } else {
                    None
                }
            })
            .expect("expected at least one empty annotation");

        assert!(sample.boxes.is_empty());
        assert!(sample.labels.is_empty());
    }

    #[test]
    fn test_ssd_dataset_with_coco() {
        // load the same test dataset that Burn uses
        let coco =
            ImageFolderDataset::new_coco_detection(COCO_JSON, COCO_IMAGES)
                .unwrap();

        let dataset = SSDDataset::<B>::new(coco, 320, 320, Default::default());

        let sample = dataset.get(0).expect("sample should exist");

        // image shape
        assert_eq!(sample.image.dims(), [3, 320, 320]);

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
