use burn::backend::wgpu::Wgpu;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::tensor::backend::Backend;

use training::dataset::{SSDBatcher, SSDDataset};
use training::target::matcher::Matcher;
use training::target::target_encoder::SSDTargetEncoder;

use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;

#[test]
fn test_ssd_full_pipeline() {
    type B = Wgpu;

    let device = <B as Backend>::Device::default();

    // load tiny COCO dataset, copied from Burn's ImageFolderDataset tests
    let coco = ImageFolderDataset::new_coco_detection(
        "tests/dataset_coco.json",
        "tests/image_folder_coco",
    )
    .unwrap();

    let dataset = SSDDataset::<B>::new(coco, 320, 320, device.clone());
    let num_classes = dataset.num_classes();

    let loader = DataLoaderBuilder::new(SSDBatcher::new())
        .batch_size(2)
        .shuffle(42)
        .set_device(device.clone())
        .build(dataset);

    // pull one batch
    let mut iter = loader.iter();
    let batch = iter.next().expect("expected at least one batch");

    let model = SSDLiteMobileNetV3::<B>::new(
        MobileNetV3Arch::Large,
        num_classes,
        &device,
    );

    let anchors = model.generate_anchors_for_input(320, 320, &device);
    let num_anchors = anchors.len();

    let matcher = Matcher::new(0.5, 0.4);
    let encoder = SSDTargetEncoder::new(matcher);

    let (tgt_classes, tgt_boxes, pos_mask) = encoder.encode_batch::<B>(
        &anchors,
        batch.boxes.clone(),
        batch.labels.clone(),
        &device,
    );

    // 9. Assert shapes
    assert_eq!(tgt_classes.dims(), [2, num_anchors]);
    assert_eq!(tgt_boxes.dims(), [2, num_anchors, 4]);
    assert_eq!(pos_mask.dims(), [2, num_anchors]);
}
