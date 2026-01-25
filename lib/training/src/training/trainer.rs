use std::sync::Arc;

use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::Ignored,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{Learner, SupervisedTraining},
};

use crate::dataset::{SSDBatch, SSDBatcher, SSDDataset};
use crate::loss::SSDLoss;
use crate::target::{matcher::Matcher, target_encoder::SSDTargetEncoder};
use crate::training::ssd_train_step::SSDTrainModel;

use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;

pub fn train<B: AutodiffBackend>(
    coco_json: &str,
    coco_images: &str,
    device: B::Device,
) {
    // ---------------------------------------------------------
    // 1. Load datasets (train + valid)
    // ---------------------------------------------------------
    let input_w = 320;
    let input_h = 320;

    // ImageFolderDataset is not Clone, so build it twice
    let inner_train =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .expect("Failed to load COCO dataset (train)");

    let inner_valid =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .expect("Failed to load COCO dataset (valid)");

    // Devices
    let device_train = device.clone();
    let device_valid =
        <<B as AutodiffBackend>::InnerBackend as Backend>::Device::default();

    // Datasets
    let dataset_train =
        SSDDataset::<B>::new(inner_train, input_w, input_h, device_train);

    let dataset_valid = SSDDataset::<<B as AutodiffBackend>::InnerBackend>::new(
        inner_valid,
        input_w,
        input_h,
        device_valid,
    );

    let num_classes = dataset_train.num_classes();

    // ---------------------------------------------------------
    // 2. Build dataloaders
    //    - train: backend B, batch SSDBatch<B>
    //    - valid: backend InnerBackend, batch SSDBatch<InnerBackend>
    // ---------------------------------------------------------
    let batcher_train = SSDBatcher::new();
    let batcher_valid = SSDBatcher::new();

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(8)
        .shuffle(469)
        .build(dataset_train);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(8)
        .shuffle(1337)
        .build(dataset_valid);

    // ---------------------------------------------------------
    // 3. Build model
    // ---------------------------------------------------------
    let model = SSDLiteMobileNetV3::<B>::new(
        MobileNetV3Arch::Large,
        num_classes,
        &device,
    );

    // ---------------------------------------------------------
    // 4. Build training wrapper
    // ---------------------------------------------------------
    let matcher = Matcher::new(0.5, 0.4);
    let encoder = SSDTargetEncoder::new(matcher);
    let loss_fn = SSDLoss::new(1.0, 1.0);

    let anchors = model.generate_anchors_for_input(input_w, input_h, &device);

    let train_model = SSDTrainModel {
        model,
        anchors: Ignored(anchors),
        encoder: Ignored(encoder),
        loss_fn: Ignored(loss_fn),
        device: Ignored(device.clone()),
    };

    // ---------------------------------------------------------
    // 5. Optimizer
    // ---------------------------------------------------------
    let optim = AdamConfig::new().init::<B, SSDTrainModel<B>>();

    // ---------------------------------------------------------
    // 6. Recorder (checkpoints)
    // ---------------------------------------------------------
    let recorder = CompactRecorder::new();

    // ---------------------------------------------------------
    // 7. Build learner (constant LR scheduler = f64)
    // ---------------------------------------------------------
    let lr: f64 = 1e-4;
    let learner = Learner::new(train_model, optim, lr);

    // ---------------------------------------------------------
    // 8. Run training
    // ---------------------------------------------------------
    let training = SupervisedTraining::new(
        "runs/ssd-experiment",
        train_loader,
        valid_loader,
    )
    .num_epochs(10)
    .with_file_checkpointer(recorder);

    let _trained = training.launch(learner);

    println!("Training complete!");
}
