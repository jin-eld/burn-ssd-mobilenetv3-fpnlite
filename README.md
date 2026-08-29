# burn-ssd-mobilenetv3-fpnlite
SSD MobileNet V3 FPNLite implementation in Burn

This project is still a work in progress, the goal is to be able to train an
SSD MobileNet FPNLite 320x320 model with an export possibility to TFLite for
final inference on the Coral Edge TPU.

## Current Status
* implemented SSD-MobilenetV3-FPNLite inference
* import of Pytorch `.pth` weights for MobileNetV3 "large" and "small"
* implemented training (for now single GPU, targetting Vulkan/WGPU)

### Missing
* training data augmentation
* multi GPU support

## Running

### Inference

There is a mobilenetv3-cli utility for testing the current code.
```
Usage: mobilenetv3-cli <image_path> [-t <model-type>]

mobilenetv3-cli command line arguments

Positional Arguments:
  image_path        file name of the image for inference

Options:
  -t, --model-type  select model type, either "large" (default) or "small"
  --help, help      display usage information
```

`cargo run -- /path/to/image.jpg`

Running the cli utility without the `pretrained` feature will use an
empty model, which is only handy during development, but has no real value
otherwise.

### Training

There is a separate command line utility for training:

```
Usage: mobilenetv3-train --coco-json <coco-json> --coco-images <coco-images>
                        [--epochs <epochs>] [--batch-size <batch-size>]
                        [--seed <seed>] [--output <output>] [--resume]

Options:
  --coco-json       path to the COCO-format JSON annotation file
  --coco-images     path to the directory containing the training images
  --epochs          number of training epochs
  --batch-size      batch size for training and validation
  --seed            random seed for data loader shuffling
  --output          save training output (checkpoints, logs, etc) to this
                    directory
  --resume          resume training from the latest checkpoint in the output
                    directory
  --help, help      display usage information
```

Example:
```
cargo run --release --bin mobilenetv3-train -- \
          --coco-json /path/to/coco/labels.json \
          --coco-images /path/to/coco/imagedata
```
