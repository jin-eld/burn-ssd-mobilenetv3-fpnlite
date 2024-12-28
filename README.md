# burn-ssd-mobilenetv3-fpnlite
SSD MobileNet V3 FPNLite implementation in Burn

This project is still a work in progress, the goal is to be able to train an SSD MobileNet FPNLite 320x320 model with an export possibility to TFLite for final inference on the Coral Edge TPU.

## Current Status
* implemented MobileNet V3 (inference only, untested)

## Known Issues
* It is currently not possible to import a pretrained model from PyTorch due to an issue with Burn
