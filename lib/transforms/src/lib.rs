use burn::prelude::*;
use image::{
    imageops::{crop_imm, FilterType},
    DynamicImage, GenericImageView, ImageBuffer, Rgb,
};

pub fn img_resize_and_center_crop(
    img: &DynamicImage,
    target_size: u32,
) -> DynamicImage {
    let (width, height) = img.dimensions();

    let (new_width, new_height) = if width < height {
        (target_size, (height * target_size) / width)
    } else {
        ((width * target_size) / height, target_size)
    };

    let resized = img.resize_exact(new_width, new_height, FilterType::Triangle);

    let crop_x = if new_width > target_size {
        (new_width / 2) - (target_size / 2)
    } else {
        0
    };

    let crop_y = if new_height > target_size {
        (new_height / 2) - (target_size / 2)
    } else {
        0
    };

    let cropped = crop_imm(
        &resized,
        crop_x.min(new_width - target_size),
        crop_y.min(new_height - target_size),
        target_size,
        target_size,
    );

    // drop the alpha channel
    let ret: DynamicImage = cropped.to_image().into();
    return ret.to_rgb8().into();
}

// Based on:
// https://github.com/tracel-ai/models/blob/main/mobilenetv2-burn/examples/inference.rs
pub fn img_to_tensor<B: Backend>(
    img: DynamicImage,
    device: &Device<B>,
) -> Tensor<B, 3> {
    let (width, height) = img.dimensions();
    let raw_pixels = img.into_rgb8().into_raw();
    Tensor::<B, 3>::from_data(TensorData::new(raw_pixels, Shape::new([height as usize, width as usize, 3])), device)
        .permute([2, 0, 1])  // [H, W, C] -> [C, H, W]
        / 255 // normalize between [0, 1]
}

pub fn tensor_to_img<B: Backend>(
    tensor: Tensor<B, 4>,
    width: u32,
    height: u32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    // Convert to [0, 1] range
    let tensor = tensor.squeeze_dims::<3>(&[0]);

    // Reverse permute: [C, H, W] -> [H, W, C]
    let tensor = tensor.permute([1, 2, 0]);

    // Convert to u8 and save the image
    let data: Vec<f32> = tensor.into_data().to_vec().unwrap();
    let data_u8: Vec<u8> = data
        .iter()
        .map(|&x| (x * 255.0).min(255.0).max(0.0) as u8)
        .collect();

    // Create the image buffer
    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width as u32, height as u32, data_u8).unwrap();
    return image_buffer;
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgba, RgbaImage};

    /// Helper function to create a sample image
    fn create_test_image(
        width: u32,
        height: u32,
        color: Rgba<u8>,
    ) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_resize_and_center_crop() {
        let img = create_test_image(100, 200, Rgba([255, 0, 0, 255])); // R

        let target_size = 50;
        let output = img_resize_and_center_crop(&img, target_size);

        assert_eq!(output.dimensions(), (target_size, target_size));
    }

    #[test]
    fn test_equal_dimension_resize() {
        let img = create_test_image(150, 150, Rgba([0, 255, 0, 255])); // G

        let target_size = 75;
        let output = img_resize_and_center_crop(&img, target_size);

        assert_eq!(output.dimensions(), (target_size, target_size));
    }

    #[test]
    fn test_already_target_size() {
        let img = create_test_image(50, 50, Rgba([0, 0, 255, 255])); // B

        let target_size = 50;
        let output = img_resize_and_center_crop(&img, target_size);

        assert_eq!(output.dimensions(), (target_size, target_size));
    }

    #[test]
    fn test_smaller_than_target() {
        let img = create_test_image(30, 40, Rgba([255, 255, 0, 255])); // A yellow image

        let target_size = 50;
        let output = img_resize_and_center_crop(&img, target_size);

        assert_eq!(output.dimensions(), (target_size, target_size));
    }
}
