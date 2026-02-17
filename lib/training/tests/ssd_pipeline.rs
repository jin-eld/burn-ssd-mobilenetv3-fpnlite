use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::tensor::Device;
use burn::train::TrainStep;

use training::dataset::{SSDBatcher, SSDDataset};
use training::loss::SSDLoss;
use training::target::matcher::Matcher;
use training::target::target_encoder::SSDTargetEncoder;
use training::training::ssd_train_step::SSDTrainModel;

use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;

#[test]
fn test_ssd_full_pipeline() {
    let device = Device::default();

    // load tiny COCO dataset, copied from Burn's ImageFolderDataset tests
    let coco = ImageFolderDataset::new_coco_detection(
        "tests/dataset_coco.json",
        "tests/image_folder_coco",
    )
    .unwrap();

    let dataset = SSDDataset::new(coco, 320, 320, device.clone());
    let num_classes = dataset.num_classes();

    let loader = DataLoaderBuilder::new(SSDBatcher::new())
        .batch_size(2)
        .shuffle(42)
        .set_device(device.clone())
        .build(dataset);

    // pull one batch
    let mut iter = loader.iter();
    let batch = iter
        .next()
        .expect("expected at least one batch")
        .expect("expected batch loading to succeed");

    let model =
        SSDLiteMobileNetV3::new(MobileNetV3Arch::Large, num_classes, &device);

    let anchors = model.generate_anchors_for_input(320, 320, &device);
    let num_anchors = anchors.len();

    let matcher = Matcher::new(0.5, 0.4);
    let encoder = SSDTargetEncoder::new(matcher);

    let (tgt_classes, tgt_boxes, pos_mask) = encoder.encode_batch(
        &anchors,
        batch.boxes.clone(),
        batch.labels.clone(),
        &device,
    );

    // assert shapes
    assert_eq!(tgt_classes.dims(), [2, num_anchors]);
    assert_eq!(tgt_boxes.dims(), [2, num_anchors, 4]);
    assert_eq!(pos_mask.dims(), [2, num_anchors]);
}

#[test]
fn training_test_forward_backward() {
    let device = Device::default().autodiff();

    let coco = ImageFolderDataset::new_coco_detection(
        "tests/dataset_coco.json",
        "tests/image_folder_coco",
    )
    .unwrap();

    let dataset = SSDDataset::new(coco, 320, 320, device.clone());

    let loader = DataLoaderBuilder::new(SSDBatcher::new())
        .batch_size(1)
        .set_device(device.clone())
        .build(dataset);

    let mut iter = loader.iter();
    let batch = iter
        .next()
        .expect("expected at least one batch")
        .expect("expected batch loading to succeed");

    let model = SSDLiteMobileNetV3::new(MobileNetV3Arch::Large, 2, &device);

    let anchors = model.generate_anchors_for_input(320, 320, &device);
    let encoder = SSDTargetEncoder::new(Matcher::new(0.5, 0.4));
    let loss_fn = SSDLoss::new(1.0, 1.0);

    let train_model = SSDTrainModel {
        model,
        anchors: anchors,
        encoder: encoder,
        loss_fn: loss_fn,
        device: device.clone(),
    };

    // TrainStep::step -> TrainOutput<SSDOutput>
    let out = TrainStep::step(&train_model, batch);

    let loss_val = out.item.loss.to_data().to_vec::<f32>().unwrap()[0];

    assert!(loss_val.is_finite());
}

#[test]
fn training_test_optimizer_step() {
    use burn::optim::AdamConfig;
    use burn::train::Learner;

    let device = Device::default().autodiff();

    let coco = ImageFolderDataset::new_coco_detection(
        "tests/dataset_coco.json",
        "tests/image_folder_coco",
    )
    .unwrap();

    let dataset = SSDDataset::new(coco, 320, 320, device.clone());
    let loader = DataLoaderBuilder::new(SSDBatcher::new())
        .batch_size(1)
        .set_device(device.clone())
        .build(dataset);

    let mut iter = loader.iter();
    let batch = iter
        .next()
        .expect("expected at least one batch")
        .expect("expected batch loading to succeed");

    let model = SSDLiteMobileNetV3::new(MobileNetV3Arch::Large, 2, &device);

    let anchors = model.generate_anchors_for_input(320, 320, &device);
    let encoder = SSDTargetEncoder::new(Matcher::new(0.5, 0.4));
    let loss_fn = SSDLoss::new(1.0, 1.0);

    let train_model = SSDTrainModel {
        model,
        anchors: anchors,
        encoder: encoder,
        loss_fn: loss_fn,
        device: device.clone(),
    };

    let optim = AdamConfig::new().init();
    let mut learner = Learner::new(train_model, optim, 1e-4_f64);

    let out = TrainStep::step(&learner.model(), batch);

    learner.optimizer_step(out.grads);
}
