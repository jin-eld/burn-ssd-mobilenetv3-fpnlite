# burn-ssd-mobilenetv3-fpnlite
SSD MobileNet V3 FPNLite implementation in Burn

This project is still a work in progress, the goal is to be able to train an
SSD MobileNet FPNLite 320x320 model with an export possibility to TFLite for
final inference on the Coral Edge TPU.

## Current Status
* implemented MobileNet V3 (inference does not yet work correctly)
* import of Pytorch `.pth` weights for MobileNetV3 "large" and "small"

## Running

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

## Known Issues
* Currently inference produces inconsistent results, most likely something is
  either missing or has not been initialized correctly; debugging in progress.
