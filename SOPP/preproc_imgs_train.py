"""
Script for pre-processing all-sky images to correct format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

os.chdir('../data')
path = os.getcwd()
print(path)

path_img_info = '/scratch/oath_v1.1/classifications/classifications.csv'
path_unprocessed_imgs = '/scratch/oath_v1.1/cropped_scaled'
path_processed_imgs = '/scratch/SOPP/processed_imgs'

# Fetch information about processing said image

df = pd.read_csv(path_img_info, header=16)
print(df)
rot_angle = np.array(df['rotAng'])

# Fetch image from path. Apply rotation. Place image in new path
png = '.png'
for img_number in range(1, rot_angle.shape[0]+1):
    filename = str(img_number).zfill(5) + '.png'
    print(filename)

# Rotation

# Scale / crop image

# Place new image in new path