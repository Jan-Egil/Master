#import torch
#import torchvision.models #import shufflenet_v2_x1_0

import torchvision
model = torchvision.models.shufflenet_v2_x1_0(weights=None)

image_path = ''
feature_path = ''

# Throw images through model

# Extract features from images

# Place features in hdf-file for extraction