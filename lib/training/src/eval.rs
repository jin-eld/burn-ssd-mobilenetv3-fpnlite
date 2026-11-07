use crate::dataset::{SSDBatcher, SSDDataset};
use crate::loss::SSDLoss;
use crate::target::{matcher::Matcher, target_encoder::SSDTargetEncoder};
use crate::training::ssd_train_step::{anchors_to_tensor, SSDTrainModel};
use crate::training::trainer::split_indices;
use ab_glyph::{FontVec, PxScale};
use burn::{
    data::dataloader::DataLoaderBuilder, data::dataset::Dataset,
    module::Module, store::ModuleRecord, tensor::Device,
};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use mobilenetv3::MobileNetV3Arch;
use ssd::model::SSDLiteMobileNetV3;
use ssd::ops::map::{evaluate_map, MapResult};
use ssd::ops::postprocess::ssd_postprocess;
use std::collections::HashMap;

pub struct VisualizeConfig {
    pub num_images: usize,
    pub output_dir: String,
    pub score_threshold: f32,
}

#[derive(serde::Deserialize)]
struct CocoCategory {
    id: usize,
    name: String,
}

#[derive(serde::Deserialize)]
struct CocoJsonSubset {
    categories: Vec<CocoCategory>,
}

/// Parses the COCO JSON to extract the category ID -> name mapping.
fn load_class_names(coco_json: &str) -> Result<HashMap<usize, String>, String> {
    let content = std::fs::read_to_string(coco_json)
        .map_err(|e| format!("Failed to read COCO JSON: {}", e))?;

    // serde_json will safely ignore the "images" and "annotations" arrays
    let parsed: CocoJsonSubset = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse COCO JSON: {}", e))?;

    let mut map = HashMap::new();
    for cat in parsed.categories {
        map.insert(cat.id, cat.name);
    }
    return Ok(map);
}

/// A generic 20-color palette to cycle through for arbitrary class counts.
const PALETTE: [[u8; 3]; 20] = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [255, 128, 0],
    [128, 255, 0],
    [0, 255, 128],
    [0, 128, 255],
    [128, 0, 255],
    [255, 0, 128],
    [128, 128, 128],
    [64, 64, 64],
];
/// Standalone mAP evaluation loop.
pub fn evaluate(
    coco_json: &str,
    coco_images: &str,
    checkpoint_path: &str,
    seed: u64,
    device: &Device,
    visualize: Option<VisualizeConfig>,
) -> Result<MapResult, String> {
    let input_w = 320;
    let input_h = 320;
    let num_foreground_classes = 11;

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

    // Dynamically load class names from the COCO JSON
    let class_names = load_class_names(coco_json)?;

    let dataset_valid =
        SSDDataset::new(inner, input_w, input_h, valid_indices.clone(), false);
    println!("Evaluating on {} validation images...", dataset_valid.len());
    println!("Loaded {} class names from JSON.", class_names.len());

    // If visualization is requested, build a separate dataset instance for it.
    // ImageFolderDataset does not implement Clone, and the DataLoader consumes
    // dataset_valid, so we need a fresh instance to fetch raw pixels later.
    let dataset_viz = if visualize.is_some() {
        let inner_viz =
            burn::data::dataset::vision::ImageFolderDataset::new_coco_detection(
                coco_json,
                coco_images,
            )
            .map_err(|e| format!("Failed to load COCO dataset for viz: {}", e))?;
        Some(SSDDataset::new(
            inner_viz,
            input_w,
            input_h,
            valid_indices.clone(),
            false,
        ))
    } else {
        None
    };

    // 2. Load checkpoint
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

    let model = wrapper.model.valid();
    let anchors_tensor = anchors_to_tensor(&anchors, device);

    // Try to load a system font for text rendering; perhaps bundle the
    // font or make this configurable?
    let font =
        std::fs::read("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
            .or_else(|_| std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf"))
            .or_else(|_| {
                std::fs::read("/usr/share/fonts/dejavu/DejaVuSans.ttf")
            })
            .ok()
            .and_then(|data| FontVec::try_from_vec(data).ok());

    if visualize.is_some() && font.is_none() {
        println!("Warning: Could not load system font (DejaVuSans.ttf). Text labels will be skipped.");
    }

    // 3. Iterate over val split
    let mut all_predictions: Vec<Vec<ssd::ops::map::Detection>> = Vec::new();
    let mut all_ground_truths: Vec<Vec<ssd::ops::map::GroundTruth>> =
        Vec::new();

    let batcher = SSDBatcher::new();
    let loader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(0)
        .set_device(device.clone())
        .build(dataset_valid);

    for (i, batch_result) in loader.iter().enumerate() {
        let batch = batch_result
            .map_err(|e| format!("Dataloader error at image {}: {:?}", i, e))?;

        let (cls_logits, bbox_deltas) = model.forward_raw(batch.images);
        let decoded_boxes =
            model.decoder().decode(bbox_deltas, anchors_tensor.clone());

        let cls_logits_2d = cls_logits.squeeze_dim::<2>(0);
        let decoded_boxes_2d = decoded_boxes.squeeze_dim::<2>(0);

        // Postprocess for mAP (low threshold to capture full PR curve)
        let detections =
            ssd_postprocess(&decoded_boxes_2d, &cls_logits_2d, 0.05, 0.5, 100)
                .map_err(|e| {
                    format!("Postprocess error at image {}: {}", i, e)
                })?;

        let map_dets: Vec<ssd::ops::map::Detection> = detections
            .iter()
            .map(|d| ssd::ops::map::Detection {
                score: d.score,
                class_id: d.class,
                box_cxcywh: d.bbox,
            })
            .collect();
        all_predictions.push(map_dets);

        let gt_boxes = &batch.boxes[0];
        let gt_labels = &batch.labels[0];

        let map_gts: Vec<ssd::ops::map::GroundTruth> = gt_boxes
            .iter()
            .zip(gt_labels.iter())
            .map(|(b, l)| {
                if *l == 0 {
                    panic!("Ground truth label must be > 0");
                }
                ssd::ops::map::GroundTruth {
                    class_id: l - 1,
                    box_cxcywh: *b,
                }
            })
            .collect();
        all_ground_truths.push(map_gts);

        // Visualization pass
        if let (Some(viz), Some(viz_dataset)) = (&visualize, &dataset_viz) {
            if i < viz.num_images {
                // Fetch the raw unnormalized U8 image directly from the viz
                // dataset
                let sample = viz_dataset.get(i).map_err(|e| {
                    format!("Failed to get sample for viz: {:?}", e)
                })?;
                let raw_pixels = sample.image_data.into_bytes().to_vec();
                let mut img = RgbImage::from_raw(input_w, input_h, raw_pixels)
                    .expect("Failed to create image from raw data");

                // Run postprocess with visualization threshold (higher to
                // avoid clutter)
                let mut viz_detections = ssd_postprocess(
                    &decoded_boxes_2d,
                    &cls_logits_2d,
                    viz.score_threshold,
                    0.5,
                    100,
                )
                .unwrap_or_default();

                // Keep only the top-scoring boxes so the image stays readable
                viz_detections.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                viz_detections.truncate(20);

                for det in &viz_detections {
                    let cx = det.bbox[0];
                    let cy = det.bbox[1];
                    let w = det.bbox[2];
                    let h = det.bbox[3];

                    let x1 = ((cx - w / 2.0) * input_w as f32)
                        .max(0.0)
                        .min(input_w as f32 - 1.0)
                        as i32;
                    let y1 = ((cy - h / 2.0) * input_h as f32)
                        .max(0.0)
                        .min(input_h as f32 - 1.0)
                        as i32;
                    let x2 = ((cx + w / 2.0) * input_w as f32)
                        .max(0.0)
                        .min(input_w as f32 - 1.0)
                        as i32;
                    let y2 = ((cy + h / 2.0) * input_h as f32)
                        .max(0.0)
                        .min(input_h as f32 - 1.0)
                        as i32;

                    let color_idx = det.class % PALETTE.len();
                    let color = Rgb(PALETTE[color_idx]);

                    let rect = Rect::at(x1, y1).of_size(
                        (x2 - x1).max(0) as u32,
                        (y2 - y1).max(0) as u32,
                    );
                    draw_hollow_rect_mut(&mut img, rect, color);

                    if let Some(ref font) = font {
                        // Map dense class ID (0-10) back to COCO category
                        // ID (1-11)
                        let coco_cat_id = det.class + 1;
                        let class_name = class_names
                            .get(&coco_cat_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");

                        let text = format!("{} {:.2}", class_name, det.score);
                        let scale = PxScale::from(20.0);
                        let text_y = (y1 - 20).max(0);
                        draw_text_mut(
                            &mut img, color, x1, text_y, scale, font, &text,
                        );
                    }
                }

                std::fs::create_dir_all(&viz.output_dir).ok();
                let out_path = format!("{}/val_{:04}.png", viz.output_dir, i);
                img.save(&out_path).map_err(|e| {
                    format!("Failed to save image {}: {}", out_path, e)
                })?;
                println!(
                    "  Saved visualization to {} ({} detections)",
                    out_path,
                    viz_detections.len()
                );
            }
        }

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
