"""
Script for pre-processing all-sky images to correct format
"""

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

path_img_info = '/scratch/oath_v1.1/classifications/classifications.csv'
path_unprocessed_imgs = '/scratch/oath_v1.1/images/cropped_scaled/'
path_processed_imgs = '/scratch/SOPP/processed_imgs/'

# Fetch information about processing said image

df = pd.read_csv(path_img_info, header=16)
print(df)
rot_angle = np.array(df['rotAng'])
print(rot_angle)

# Fetch image from path. Apply rotation. Place image in new path
png = '.png'
for i in tqdm(range(rot_angle.shape[0])):
    filename = str(i+1).zfill(5) + '.png'
    full_path_fetch = path_unprocessed_imgs + filename
    full_path_save = path_processed_imgs + filename
    with Image.open(full_path_fetch) as im:
        im2 = im.rotate(-rot_angle[i])
        im2.save(full_path_save, 'png')
        del im2