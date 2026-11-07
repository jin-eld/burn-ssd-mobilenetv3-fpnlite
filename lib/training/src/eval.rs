use crate::dataset::{SSDBatcher, SSDDataset};
use crate::loss::SSDLoss;
use crate::target::{matcher::Matcher, target_encoder::SSDTargetEncoder};
use crate::training::ssd_train_step::{anchors_to_tensor, SSDTrainModel};
use crate::training::trainer::split_indices;
use burn::{
    data::dataloader::DataLoaderBuilder, data::dataset::Dataset,
    module::Module, store::ModuleRecord, tensor::Device,
};
use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;
use ssd::ops::map::{evaluate_map, MapResult};
use ssd::ops::postprocess::ssd_postprocess;

/// Standalone mAP evaluation loop.
///
/// Loads a trained checkpoint, reproduces the exact same seeded validation
/// split used during training, and computes COCO mAP metrics.
pub fn evaluate(
    coco_json: &str,
    coco_images: &str,
    checkpoint_path: &str,
    seed: u64,
    device: &Device,
) -> Result<MapResult, String> {
    let input_w = 320;
    let input_h = 320;
    let num_foreground_classes = 11; // COCO subset has 11 foreground classes (1-11)

    // 1. Load dataset and reproduce the exact same val split
    let inner =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .map_err(|e| format!("Failed to load COCO dataset: {}", e))?;

    let total_size = inner.len();
    let (_, valid_indices) = split_indices(total_size, seed);
    let total_val_images = valid_indices.len();

    // We need the model's num_classes (which includes background = 12).
    // Scanning the full dataset ensures we don't miss the highest class ID
    // if the validation split happens to exclude it.
    let all_indices: Vec<usize> = (0..total_size).collect();
    let temp_inner =
        burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
            coco_json,
            coco_images,
        )
        .map_err(|e| {
            format!("Failed to load COCO dataset for scanning: {}", e)
        })?;
    let temp_dataset =
        SSDDataset::new(temp_inner, input_w, input_h, all_indices, false);
    let num_classes_with_bg = temp_dataset.num_classes();

    let dataset_valid =
        SSDDataset::new(inner, input_w, input_h, valid_indices, false);
    println!("Evaluating on {} validation images...", dataset_valid.len());

    // 2. Load checkpoint
    // Checkpoints were saved from SSDTrainModel, so record keys are nested
    // under its `model.` field. We reconstruct the same wrapper shape to
    // load the record, then extract the inner model.
    let model = SSDLiteMobileNetV3::new(
        MobileNetV3Arch::Large,
        num_classes_with_bg,
        device,
    );
    let anchors = model.generate_anchors_for_input(input_h, input_w, device);
    let encoder = SSDTargetEncoder::new(Matcher::new(0.5, 0.4));
    let loss_fn = SSDLoss::new(1.0, 1.0);

    let wrapper = SSDTrainModel {
        model,
        anchors: anchors.clone(),
        encoder,
        loss_fn,
    };

    let record = ModuleRecord::load(checkpoint_path).map_err(|e| {
        format!("Failed to load record from {}: {:?}", checkpoint_path, e)
    })?;

    let wrapper = wrapper
        .try_load_record(record)
        .map_err(|e| format!("Failed to apply record: {:?}", e))?;

    // Switch to eval mode (disables BatchNorm training flags)
    let model = wrapper.model.valid();

    // Precompute anchors tensor once
    let anchors_tensor = anchors_to_tensor(&anchors, device);

    // 3. Iterate over val split
    let mut all_predictions: Vec<Vec<ssd::ops::map::Detection>> = Vec::new();
    let mut all_ground_truths: Vec<Vec<ssd::ops::map::GroundTruth>> =
        Vec::new();

    let batcher = SSDBatcher::new();
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(0) // deterministic order
        .set_device(device.clone())
        .build(dataset_valid);

    for (i, batch_result) in loader.iter().enumerate() {
        let batch = batch_result
            .map_err(|e| format!("Dataloader error at image {}: {:?}", i, e))?;

        // Forward pass (raw logits + deltas)
        let (cls_logits, bbox_deltas) = model.forward_raw(batch.images);

        // Decode deltas -> boxes
        let decoded_boxes =
            model.decoder().decode(bbox_deltas, anchors_tensor.clone());

        // Squeeze batch dim
        let cls_logits_2d = cls_logits.squeeze_dim::<2>(0);
        let decoded_boxes_2d = decoded_boxes.squeeze_dim::<2>(0);

        // Postprocess (sigmoid is handled inside ssd_postprocess)
        let detections = ssd_postprocess(
            &decoded_boxes_2d,
            &cls_logits_2d,
            0.05, // score threshold (low for mAP curve)
            0.5,  // IoU threshold
            100,  // max detections
        )
        .map_err(|e| format!("Postprocess error at image {}: {}", i, e))?;

        // Convert to map::Detection
        let map_dets: Vec<ssd::ops::map::Detection> = detections
            .into_iter()
            .map(|d| ssd::ops::map::Detection {
                score: d.score,
                class_id: d.class,
                box_cxcywh: d.bbox,
            })
            .collect();
        all_predictions.push(map_dets);

        // Collect GT (batch size is 1, so we take index 0)
        let gt_boxes = &batch.boxes[0];
        let gt_labels = &batch.labels[0];

        let map_gts: Vec<ssd::ops::map::GroundTruth> = gt_boxes
            .iter()
            .zip(gt_labels.iter())
            .map(|(b, l)| {
                if *l == 0 {
                    panic!("Ground truth label must be > 0 (0 is reserved for background)");
                }
                ssd::ops::map::GroundTruth {
                    class_id: l - 1, // map COCO 1-11 to dense 0-10
                    box_cxcywh: *b,
                }
            })
            .collect();
        all_ground_truths.push(map_gts);

        if (i + 1) % 50 == 0 {
            println!("  Processed {} / {} images...", i + 1, total_val_images);
        }
    }

    // 4. Compute mAP
    let map_result = evaluate_map(
        &all_predictions,
        &all_ground_truths,
        num_foreground_classes,
    );

    println!("\n=== mAP Results ===");
    println!("mAP @ 0.50:0.95 : {:.4}", map_result.map);
    println!("mAP @ 0.50      : {:.4}", map_result.map50);
    println!("mAP @ 0.75      : {:.4}", map_result.map75);
    println!("===================");

    return Ok(map_result);
}
