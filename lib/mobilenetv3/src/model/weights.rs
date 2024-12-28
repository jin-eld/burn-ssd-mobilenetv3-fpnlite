// Following the same approach as
// https://github.com/tracel-ai/models/blob/main/mobilenetv2-burn/src/model/weights.rs

use serde::{Deserialize, Serialize};

/// Pre-trained weights metadata.
pub struct Weights {
    pub(super) url: &'static str,
    pub(super) num_classes: usize,
}

const TORCH_MOBILENET_V3_NUM_CLASSES: usize = 1000;

#[cfg(feature = "pretrained")]
mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;

    impl Weights {
        /// Download the pre-trained weights to the local cache directory.
        pub fn download(&self) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("mobilenetv3-burn");

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = self.url.rsplit_once('/').unwrap().1;
            let file_name = model_dir.join(file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(
                    self.url,
                    file_base_name,
                );

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                let bytes_written = output_file.write(&bytes)?;

                if bytes_written != bytes.len() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Failed to write the whole model weights file.",
                    ));
                }
            }

            return Ok(file_name);
        }
    }
}

pub trait WeightsMeta {
    fn weights(&self) -> Weights;
}

/// Currently suported MobileNetV3 pre-trained weight types
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum MobileNetV3 {
    PyTorchLarge,
    PyTorchSmall,
}

impl WeightsMeta for MobileNetV3 {
    fn weights(&self) -> Weights {
        let url = match *self {
            MobileNetV3::PyTorchLarge => {
                "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
            },
            MobileNetV3::PyTorchSmall => {
                "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
            },
        };
        Weights {
            url,
            num_classes: TORCH_MOBILENET_V3_NUM_CLASSES,
        }
    }
}
