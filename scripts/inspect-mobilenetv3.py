#!/bin/env python3

import sys
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} large|small /path/to/model.pth")
    sys.exit(1)

# Large or Small version
model_type = sys.argv[1]
# Path to the pre-downloaded model
model_path = sys.argv[2]

import torch
import torchvision.models as models

if model_type == "large":
    model = models.mobilenet_v3_large()
elif model_type == "small":
    model = models.mobilenet_v3_small()
else:
    print(f"Unknown model type {model_type}, use either 'large' or 'small'")
    sys.exit(1)

# Load the model weights
model.load_state_dict(torch.load(model_path))

# Print layer names
for name, param in model.named_parameters():
    print(name)

