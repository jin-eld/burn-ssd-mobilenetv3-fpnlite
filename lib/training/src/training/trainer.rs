use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    tensor::Device,
    train::{Learner, SupervisedTraining},
};

use crate::dataset::{SSDBatcher, SSDDataset};
use crate::loss::SSDLoss;
use crate::target::{matcher::Matcher, target_encoder::SSDTargetEncoder};
use crate::training::ssd_train_step::SSDTrainModel;

use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;

pub fn train(coco_json: &str, coco_images: &str, device: &Device) {
    let device = device.clone().autodiff();

    let input_w = 320;
    let input_h = 320;

    // ImageFolderDataset is not Clone, need to build it twice
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

    let device_train = device.clone();
    let device_valid = device.clone();

    let dataset_train =
        SSDDataset::new(inner_train, input_w, input_h, device_train);

    let dataset_valid =
        SSDDataset::new(inner_valid, input_w, input_h, device_valid);

    let num_classes = dataset_train.num_classes();

    let batcher_train = SSDBatcher::new();
    let batcher_valid = SSDBatcher::new();

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(8)
        .shuffle(469)
        .set_device(device.clone())
        .build(dataset_train);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(8)
        .shuffle(1337)
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
        device: device.clone(),
    };

    let optim = AdamConfig::new().init();

    let lr: f64 = 1e-4;
    let learner = Learner::new(train_model, optim, lr);

    let training = SupervisedTraining::new(
        "runs/ssd-experiment",
        train_loader,
        valid_loader,
    )
    .num_epochs(10)
    .with_default_checkpointers();

    let _trained = training.launch(learner);

    println!("Training complete!");
}
