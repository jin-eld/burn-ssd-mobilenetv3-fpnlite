use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Device, Tensor};

use super::ssd_dataset::SSDSample;

#[derive(Clone, Debug)]
pub struct SSDBatch {
    pub images: Tensor<4>,         // [batch, C, H, W]
    pub boxes: Vec<Vec<[f32; 4]>>, // per-image boxes
    pub labels: Vec<Vec<usize>>,   // per-image labels
}

#[derive(Clone, Debug)]
pub struct SSDBatcher {}

impl SSDBatcher {
    pub fn new() -> Self {
        return Self {};
    }
}

impl Batcher<SSDSample, SSDBatch> for SSDBatcher {
    fn batch(&self, items: Vec<SSDSample>, device: &Device) -> SSDBatch {
        // create tensors on the target device (handles autodiff/non-autodiff seamlessly)
        let images: Vec<Tensor<3>> = items
            .iter()
            .map(|s| {
                Tensor::<3>::from_data(s.image_data.clone(), device)
                    .permute([2, 0, 1]) // [H, W, C] -> [C, H, W]
                    / 255.0 // normalize to [0, 1] and cast to float
            })
            .collect();

        // stack into [batch, C, H, W]
        let images = Tensor::stack::<4>(images, 0);

        // collect boxes and labels
        let boxes: Vec<Vec<[f32; 4]>> =
            items.iter().map(|s| s.boxes.clone()).collect();

        let labels: Vec<Vec<usize>> =
            items.iter().map(|s| s.labels.clone()).collect();

        return SSDBatch {
            images,
            boxes,
            labels,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Shape, TensorData};

    #[test]
    fn test_ssd_batcher_basic() {
        let device = Device::default();

        // two synthetic samples
        let sample1 = SSDSample {
            image_data: TensorData::new(
                vec![0u8; 320 * 320 * 3],
                Shape::new([320, 320, 3]),
            ),
            boxes: vec![[0.1, 0.2, 0.3, 0.4]],
            labels: vec![1],
        };

        let sample2 = SSDSample {
            image_data: TensorData::new(
                vec![0u8; 320 * 320 * 3],
                Shape::new([320, 320, 3]),
            ),
            boxes: vec![[0.5, 0.6, 0.7, 0.8]],
            labels: vec![0],
        };
        let batcher = SSDBatcher::new();
        let batch = batcher.batch(vec![sample1, sample2], &device);

        // image tensor shape
        assert_eq!(batch.images.dims(), [2, 3, 320, 320]);

        // boxes preserved
        assert_eq!(batch.boxes.len(), 2);
        assert_eq!(batch.boxes[0], vec![[0.1, 0.2, 0.3, 0.4]]);
        assert_eq!(batch.boxes[1], vec![[0.5, 0.6, 0.7, 0.8]]);

        // labels preserved
        assert_eq!(batch.labels.len(), 2);
        assert_eq!(batch.labels[0], vec![1]);
        assert_eq!(batch.labels[1], vec![0]);
    }

    #[test]
    fn test_ssd_batcher_empty_annotations() {
        let device = Device::default();
        let sample = SSDSample {
            image_data: TensorData::new(
                vec![0u8; 320 * 320 * 3],
                Shape::new([320, 320, 3]),
            ),
            boxes: vec![],
            labels: vec![],
        };

        let batcher = SSDBatcher::new();
        let batch = batcher.batch(vec![sample], &device);

        assert_eq!(batch.images.dims(), [1, 3, 320, 320]);
        assert!(batch.boxes[0].is_empty());
        assert!(batch.labels[0].is_empty());
    }
}
