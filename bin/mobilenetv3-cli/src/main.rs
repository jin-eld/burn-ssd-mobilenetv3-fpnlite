use argh::FromArgs;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::{
    activation::softmax, backend::Backend, cast::ToElement, Tensor,
};
use mobilenetv3::imagenet::{Normalizer, CLASSES, IMAGE_SIZE};
use std::process;
use transforms;

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
    // apply softmax to convert logits to probabilities
    let sm = softmax(output, 1);
    let (score, idx) = sm.max_dim_with_indices(1);
    let idx = idx.into_scalar().to_usize();
    let score = score.into_scalar();

    println!("Category ID: {}", idx);
    println!("Predicted Class: {}", CLASSES[idx]);
    println!("Confidence Score: {}", score);
}

fn load_and_preprocess_image<B: Backend>(
    image_path: &str,
    target_size: u32,
    device: &Device<B>,
) -> Tensor<B, 4> {
    let img = match image::open(&image_path) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("Failed to load image {}.\nError: {}", image_path, err);
            process::exit(1);
        }
    };

    let processed = transforms::img_resize_and_center_crop(&img, target_size);
    let img_tensor =
        transforms::img_to_tensor(processed, device).unsqueeze::<4>();
    return Normalizer::new(device).normalize(img_tensor);
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

    let input = load_and_preprocess_image::<MyBackend>(
        &args.image_path,
        IMAGE_SIZE,
        &device,
    );

    let output = model.forward(input);
    print_top_prediction(output);
}
