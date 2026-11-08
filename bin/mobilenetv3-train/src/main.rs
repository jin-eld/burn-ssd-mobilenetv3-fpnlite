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

    /// number of validation images to visualize with bounding boxes
    /// (requires --eval)
    #[argh(option)]
    visualize: Option<usize>,

    /// directory to save visualized images
    #[argh(option, default = "String::from(\"visualizations\")")]
    visualize_dir: String,

    /// score threshold for drawing boxes (default 0.3 to avoid clutter)
    #[argh(option, default = "0.3")]
    visualize_threshold: f32,

    /// initialize the backbone with ImageNet weights (PyTorch
    /// MobileNetV3-Large)
    #[cfg(feature = "pretrained")]
    #[argh(switch)]
    pretrained_backbone: bool,
}

fn main() {
    let args: Args = argh::from_env();

    if args.eval {
        let device = Device::wgpu(Default::default()); // no autodiff for eval
        let checkpoint_path = args.checkpoint.unwrap_or_else(|| {
            eprintln!("Error: --checkpoint is required when using --eval");
            std::process::exit(1);
        });

        let viz_config =
            args.visualize.map(|n| training::eval::VisualizeConfig {
                num_images: n,
                output_dir: args.visualize_dir.clone(),
                score_threshold: args.visualize_threshold,
            });

        if let Err(e) = training::eval::evaluate(
            &args.coco_json,
            &args.coco_images,
            &checkpoint_path,
            args.seed,
            &device,
            viz_config,
        ) {
            eprintln!("Evaluation failed: {}", e);
            std::process::exit(1);
        }
        return;
    }

    let device = Device::wgpu(Default::default()).autodiff();

    #[cfg(not(feature = "pretrained"))]
    let pretrained_backbone = false;
    #[cfg(feature = "pretrained")]
    let pretrained_backbone = args.pretrained_backbone;

    if let Err(e) = training::training::trainer::train(
        &args.coco_json,
        &args.coco_images,
        args.epochs,
        args.batch_size,
        args.seed,
        &args.output,
        args.resume,
        pretrained_backbone,
        &device,
    ) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
