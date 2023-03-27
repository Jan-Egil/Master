"""
Script for pre-processing all-sky images to correct format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_img_info = '/scratch/oath_v1.1/classifications/classifications.csv'
path_unprocessed_imgs = '/scratch/oath_v1.1/cropped_scaled'
path_processed_imgs = '/scratch/SOPP/processed_imgs'

# Fetch information about processing said image

df = pd.read_csv(path_img_info, header=18)
print(df)

# Fetch image from path

# Rotation

# Scale / crop image

# Place new image in new path