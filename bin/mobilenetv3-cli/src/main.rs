use argh::FromArgs;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{
    backend::Backend, cast::ToElement, Element, Tensor, TensorData,
};
use mobilenetv3::imagenet::{Normalizer, CLASSES, HEIGHT, WIDTH};
use std::process;

#[cfg(not(feature = "pretrained"))]
use mobilenetv3::MobileNetV3Config;

#[cfg(feature = "pretrained")]
use mobilenetv3::{weights, MobileNetV3PretrainedConfig};

type MyBackend = Wgpu<f32, i32>;

#[derive(FromArgs)]
/// mobilenetv3-cli command line arguments
struct Arguments {
    /// select model type, either "large" (default) or "small"
    #[argh(option, short = 't')]
    model_type: Option<String>,

    /// file name of the image for inference
    #[argh(positional)]
    image_path: String,
}

fn print_top_prediction<B: Backend>(output: Tensor<B, 2>) {
    let (score, idx) = output.max_dim_with_indices(1);
    let idx = idx.into_scalar().to_usize();

    println!(
        "Predicted: {}\nCategory Id: {}\nScore: {:.4}",
        CLASSES[idx],
        idx,
        score.into_scalar()
    );
}

// From https://github.com/tracel-ai/models/blob/main/mobilenetv2-burn/examples/inference.rs
fn to_tensor<B: Backend, T: Element>(
    data: Vec<T>,
    shape: [usize; 3],
    device: &Device<B>,
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), device)
        // [H, W, C] -> [C, H, W]
        .permute([2, 0, 1])
        / 255 // normalize between [0, 1]
}

fn load_and_preprocess_image<B: Backend>(
    image_path: &str,
    width: usize,
    height: usize,
    device: &Device<B>,
) -> Tensor<B, 4> {
    let img = match image::open(&image_path) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("Failed to load image {}.\nError: {}", image_path, err);
            process::exit(1);
        }
    };

    let resized = img.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    // Create tensor from image data
    let img_tensor =
        to_tensor(resized.into_rgb8().into_raw(), [height, width, 3], device)
            .unsqueeze::<4>(); // [B, C, H, W]

    return img_tensor;
}

fn main() {
    let args: Arguments = argh::from_env();

    let device = burn::backend::wgpu::WgpuDevice::default();
    let model: mobilenetv3::MobileNetV3<MyBackend>;

    #[cfg(feature = "pretrained")]
    {
        let weights = match args.model_type.as_deref() {
            Some("large") => weights::MobileNetV3::PyTorchLarge,
            Some("small") => weights::MobileNetV3::PyTorchSmall,
            Some(x) => {
                eprintln!("Invalid model type {}", x);
                std::process::exit(1);
            }
            None => weights::MobileNetV3::PyTorchLarge, // default
        };

        model = match MobileNetV3PretrainedConfig::new(weights).init(&device) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };
    }

    #[cfg(not(feature = "pretrained"))]
    {
        println!(
            "Warning, you are using an empty model, dev testing use case only!"
        );

        let config = MobileNetV3Config::new().with_num_classes(CLASSES.len());
        model = match args.model_type.as_deref() {
            Some("large") => config.init_large(&device),
            Some("small") => config.init_small(&device),
            Some(x) => {
                eprintln!("Invalid model type {}", x);
                std::process::exit(1);
            }
            None => config.init_large(&device),
        };
    }

    // Load and preprocess the actual image
    let input = load_and_preprocess_image::<MyBackend>(
        &args.image_path,
        WIDTH,
        HEIGHT,
        &device,
    );

    let input = Normalizer::new(&device).normalize(input);

    let output = model.forward(input);
    print_top_prediction(output);
}
