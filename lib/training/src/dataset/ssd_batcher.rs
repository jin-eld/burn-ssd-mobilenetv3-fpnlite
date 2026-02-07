use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Tensor};

use super::ssd_dataset::SSDSample;

#[derive(Clone, Debug)]
pub struct SSDBatch<B: Backend> {
    pub images: Tensor<B, 4>,      // [batch, C, H, W]
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

impl<B: Backend> Batcher<B, SSDSample<B>, SSDBatch<B>> for SSDBatcher {
    fn batch(
        &self,
        items: Vec<SSDSample<B>>,
        device: &B::Device,
    ) -> SSDBatch<B> {
        // move images to device and collect them
        let images: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|s| s.image.clone().to_device(device))
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
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{backend::Backend, Tensor};

    type B = Wgpu;

    #[test]
    fn test_ssd_batcher_basic() {
        let device = <B as Backend>::Device::default();

        // two synthetic samples
        let sample1 = SSDSample {
            image: Tensor::<B, 3>::zeros([3, 320, 320], &device),
            boxes: vec![[0.1, 0.2, 0.3, 0.4]],
            labels: vec![1],
        };

        let sample2 = SSDSample {
            image: Tensor::<B, 3>::zeros([3, 320, 320], &device),
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
        let device = <B as Backend>::Device::default();

        let sample = SSDSample {
            image: Tensor::<B, 3>::zeros([3, 320, 320], &device),
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
