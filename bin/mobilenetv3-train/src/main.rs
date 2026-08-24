use argh::FromArgs;
use burn::tensor::Device;

/// SSD-MobileNetV3-FPNLite training CLI.
#[derive(FromArgs)]
struct Args {
    /// path to the COCO-format JSON annotation file
    #[argh(option)]
    coco_json: String,

    /// path to the directory containing the training images
    #[argh(option)]
    coco_images: String,

    /// number of training epochs
    #[argh(option, default = "10")]
    epochs: usize,

    /// batch size for training and validation
    #[argh(option, default = "8")]
    batch_size: usize,

    /// random seed for data loader shuffling
    #[argh(option, default = "469")]
    seed: u64,

    /// save training output (checkpoints, logs, etc) to this directory
    #[argh(option, default = "String::from(\"training-output\")")]
    output: String,

    /// resume training from the latest checkpoint in the output directory
    #[argh(switch)]
    resume: bool,
}

fn main() {
    let args: Args = argh::from_env();

    let device = Device::default().autodiff();

    if let Err(e) = training::training::trainer::train(
        &args.coco_json,
        &args.coco_images,
        args.epochs,
        args.batch_size,
        args.seed,
        &args.output,
        args.resume,
        &device,
    ) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
