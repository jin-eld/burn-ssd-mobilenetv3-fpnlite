#![recursion_limit = "256"] // as suggested by burn-wgpu docs
use argh::FromArgs;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{activation::softmax, Device, Float, Tensor};
use burn_dispatch::DispatchDevice;

use mobilenetv3::imagenet::{Normalizer, CLASSES, IMAGE_SIZE};
use std::process;
use transforms;

#[cfg(not(feature = "pretrained"))]
use mobilenetv3::MobileNetV3Config;

#[cfg(feature = "pretrained")]
use mobilenetv3::{weights, MobileNetV3PretrainedConfig};

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

fn print_top_prediction(output: Tensor<2, Float>) {
    // apply softmax to convert logits to probabilities
    let sm = softmax(output, 1);

    let score_tensor = sm.clone().max_dim(1);
    let idx_tensor = sm.argmax(1);

    let idx = idx_tensor.into_scalar::<i64>() as usize;
    let score = score_tensor.into_scalar::<f32>();

    println!("Category ID: {}", idx);
    println!("Predicted Class: {}", CLASSES[idx]);
    println!("Confidence Score: {}", score);
}

fn load_and_preprocess_image(
    image_path: &str,
    target_size: u32,
    device: &Device,
) -> Tensor<4> {
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

    let dispatch_device = DispatchDevice::Wgpu(WgpuDevice::default());
    let device: Device = dispatch_device.into();

    let model: mobilenetv3::MobileNetV3;

    #[cfg(feature = "pretrained")]
    {
        // Bring the extension trait into scope so .load_from becomes available
        use burn_store::ModuleSnapshot;

        let weights_type = match args.model_type.as_deref() {
            Some("large") => weights::MobileNetV3::PyTorchLarge,
            Some("small") => weights::MobileNetV3::PyTorchSmall,
            Some(x) => {
                eprintln!("Invalid model type {}", x);
                std::process::exit(1);
            }
            None => weights::MobileNetV3::PyTorchLarge, // default
        };

        // Destructure the tuple returned by your Config's init method
        // Note: Since device properties changed in 0.22, we pass device.clone()
        let (mut pretrained_model, mut store) =
            MobileNetV3PretrainedConfig::new(weights_type)
                .init(&device.clone())
                .unwrap_or_else(|e| {
                    eprintln!("Failed to load model config or weights: {}", e);
                    std::process::exit(1);
                });

        // Hydrate the weights directly into your newly configured model instance
        pretrained_model
            .load_from(&mut store)
            .expect("Failed to load PyTorch model weights via burn-store");

        // Assign back to your working model variable
        model = pretrained_model;
    }

    #[cfg(not(feature = "pretrained"))]
    {
        use mobilenetv3::MobileNetV3Config;
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

    let input =
        load_and_preprocess_image(&args.image_path, IMAGE_SIZE, &device);

    let output = model.forward(input);
    print_top_prediction(output);
}
