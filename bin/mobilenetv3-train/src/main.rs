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

    /// evaluate a checkpoint instead of training
    #[argh(switch)]
    eval: bool,

    /// path to the checkpoint file to evaluate (required if --eval is set)
    #[argh(option)]
    checkpoint: Option<String>,
}

fn main() {
    let args: Args = argh::from_env();

    if args.eval {
        let device = Device::wgpu(Default::default()); // no autodiff for eval
        let checkpoint_path = args.checkpoint.unwrap_or_else(|| {
            eprintln!("Error: --checkpoint is required when using --eval");
            std::process::exit(1);
        });

        if let Err(e) = training::eval::evaluate(
            &args.coco_json,
            &args.coco_images,
            &checkpoint_path,
            args.seed,
            &device,
        ) {
            eprintln!("Evaluation failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let device = Device::wgpu(Default::default()).autodiff();

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
