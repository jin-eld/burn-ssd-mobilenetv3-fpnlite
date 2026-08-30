use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::{lr_scheduler::LrSchedulerRecord, AdamConfig},
    store::ModuleRecord,
    tensor::Device,
    train::{metric::LossMetric, Learner, SupervisedTraining},
};
use std::path::{Path, PathBuf};

use crate::dataset::{SSDBatcher, SSDDataset};
use crate::loss::SSDLoss;
use crate::target::{matcher::Matcher, target_encoder::SSDTargetEncoder};
use crate::training::ssd_output::{
    AvgScalarMetric, ClsLossInput, PosCountInput, RegLossInput,
};
use crate::training::ssd_train_step::SSDTrainModel;

use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;

// Scans the checkpoint directory for `model-*.bpk` files and returns the
// highest epoch number.
// Returns `Ok(None)` if the directory exists but contains no valid checkpoints.
fn find_latest_checkpoint(
    checkpoint_dir: &Path,
) -> Result<Option<usize>, String> {
    let mut latest = 0;
    let entries = std::fs::read_dir(checkpoint_dir).map_err(|e| {
        format!(
            "Failed to read checkpoint directory {:?}: {}",
            checkpoint_dir, e
        )
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("model-") && name.ends_with(".bpk") {
                if let Some(num_str) = name
                    .strip_prefix("model-")
                    .and_then(|s| s.strip_suffix(".bpk"))
                {
                    if let Ok(epoch) = num_str.parse::<usize>() {
                        latest = latest.max(epoch);
                    }
                }
            }
        }
    }

    if latest == 0 {
        return Ok(None);
    }
    return Ok(Some(latest));
}

pub fn train(
    coco_json: &str,
    coco_images: &str,
    epochs: usize,
    batch_size: usize,
    seed: u64,
    output: &str,
    resume: bool,
    device: &Device,
) -> Result<(), String> {
    let input_w = 320;
    let input_h = 320;

    // ImageFolderDataset is not Clone, need to build it twice
    let inner_train =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .map_err(|e| {
            format!("Failed to load COCO dataset (training): {}", e)
        })?;

    let inner_valid =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .map_err(|e| {
            format!("Failed to load COCO dataset (validation): {}", e)
        })?;

    let dataset_train = SSDDataset::new(inner_train, input_w, input_h);
    let dataset_valid = SSDDataset::new(inner_valid, input_w, input_h);

    let num_classes = dataset_train.num_classes();

    let batcher_train = SSDBatcher::new();
    let batcher_valid = SSDBatcher::new();

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(seed)
        .set_device(device.clone())
        .build(dataset_train);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .shuffle(seed + 1)
        .set_device(device.clone())
        .build(dataset_valid);

    let model =
        SSDLiteMobileNetV3::new(MobileNetV3Arch::Large, num_classes, &device);

    let matcher = Matcher::new(0.5, 0.4);
    let encoder = SSDTargetEncoder::new(matcher);
    let loss_fn = SSDLoss::new(1.0, 1.0);

    let anchors = model.generate_anchors_for_input(input_w, input_h, &device);

    let train_model = SSDTrainModel {
        model,
        anchors: anchors,
        encoder: encoder,
        loss_fn: loss_fn,
    };

    let checkpoint_dir = PathBuf::from(output).join("checkpoint");
    let mut start_epoch = 0;
    let mut optim = AdamConfig::new().init();

    if resume {
        if !checkpoint_dir.exists() {
            return Err(format!(
                "Cannot resume: checkpoint directory {:?} does not exist. Did you pass the wrong --output directory?",
                checkpoint_dir
            ));
        }

        match find_latest_checkpoint(&checkpoint_dir)? {
            Some(epoch) => {
                println!("Resuming from epoch {}", epoch);
                start_epoch = epoch;

                let optim_path =
                    checkpoint_dir.join(format!("optim-{}.bpk", epoch));
                optim = optim.load(&optim_path).map_err(|e| {
                    format!(
                        "Failed to load optimizer record from {:?}: {:?}",
                        optim_path, e
                    )
                })?;
            }
            None => {
                return Err(format!(
                    "Cannot resume: no checkpoint files found in {:?}",
                    checkpoint_dir
                ));
            }
        }
    }

    let lr: f64 = 1e-4;
    let mut learner = Learner::new(train_model, optim, lr);

    // load existing checkpoints for resuming
    if start_epoch > 0 {
        let model_path =
            checkpoint_dir.join(format!("model-{}.bpk", start_epoch));
        let sched_path =
            checkpoint_dir.join(format!("scheduler-{}.bpk", start_epoch));

        let model_record = ModuleRecord::load(&model_path).map_err(|e| {
            format!(
                "Failed to load model record from {:?}: {:?}",
                model_path, e
            )
        })?;
        let sched_record =
            LrSchedulerRecord::load(&sched_path).map_err(|e| {
                format!(
                    "Failed to load scheduler record from {:?}: {:?}",
                    sched_path, e
                )
            })?;

        learner.load_model(model_record);
        learner.load_scheduler(sched_record);
    }

    let mut training =
        SupervisedTraining::new(output, train_loader, valid_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(AvgScalarMetric::<ClsLossInput>::new(
                "Loss Cls",
            ))
            .metric_valid_numeric(AvgScalarMetric::<ClsLossInput>::new(
                "Loss Cls",
            ))
            .metric_train_numeric(AvgScalarMetric::<RegLossInput>::new(
                "Loss Reg",
            ))
            .metric_valid_numeric(AvgScalarMetric::<RegLossInput>::new(
                "Loss Reg",
            ))
            .metric_train_numeric(AvgScalarMetric::<PosCountInput>::new(
                "Pos Anchors",
            ))
            .metric_valid_numeric(AvgScalarMetric::<PosCountInput>::new(
                "Pos Anchors",
            ))
            .num_epochs(epochs)
            .with_default_checkpointers();

    // Tell the training loop to start counting from the resumed epoch
    if start_epoch > 0 {
        training = training.checkpoint(start_epoch);
    }

    let training = training.summary();

    let _trained = training.launch(learner);

    println!("Training complete!");

    return Ok(());
}
