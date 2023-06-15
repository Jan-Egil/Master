import h5py
import pandas as pd
import numpy as np
from sys import platform
from PIL import Image
import os

import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

if platform == 'win32':
    os.environ['CDF_LIB'] = "C:\\Users\janeg\Desktop\CDF\\bin"
    cdfpath = "C:\\Users\janeg\Desktop\picfiles\\thg_l1_asf_gill_2012100508_v01.cdf"
    curr_path = os.getcwd()
    temp_dir = "\\tempdir\\"
    full_path = curr_path+temp_dir
else:
    os.environ["CDF_LIB"] = "/usr/bin/"#"/uio/hume/student-u58/janeod/.conda/envs/SVM/lib/python3.10/site-packages/spacepy"
    cdfpath = ...

from spacepy import pycdf

cdf_path = full_path + "temp2.cdf"
cdf = pycdf.CDF(cdf_path)

#loc = "gill"
loc = 'fsim'

key_img = f"thg_asf_{loc}"

num_imgs = cdf[key_img].shape[0]
print(num_imgs)

#print(cdf[f"{key_img}"][...])

img_array = cdf[key_img][660]

# Crop the image

crop_percent = 12
num_pixels = int(img_array.shape[0]*(crop_percent/100))
temp_img = Image.fromarray(img_array)
(left, upper, right, lower) = (num_pixels, num_pixels, img_array.shape[0]-num_pixels, img_array.shape[0]-num_pixels)
temp_img2 = temp_img.crop((left, upper, right, lower))
img_array = np.asarray(temp_img2)



# Rescale to range [0,1]
# 1st percentile, subtract this
percentile1 = np.percentile(img_array, 1)
img_array = img_array - percentile1

# Then 99th percentile (of altered array), divide by this.
percentile99 = np.percentile(img_array, 99)
img_array = img_array/percentile99

# Then set everything under 0 to 0, and everything over 1 to 1.
img_array[img_array > 1] = 1
img_array[img_array < 0] = 0

"""
rgb_filter = np.zeros_like(img_array)
img_array = np.array([rgb_filter, img_array*255, rgb_filter])
print(img_array)
"""

img = Image.fromarray(img_array*255)
img = img.resize((256,256))
img.show()

cdf.close()


"""
path_feats_dir = '/scratch/asim/'
path_feats_list = ['201011.hdf', '201012.hdf', '201101.hdf', '201102.hdf']
path_feats = path_feats_dir + path_feats_list[0]

df2 = pd.read_hdf(path_feats)
print(df2)
"""
"""
with h5py.File(path_feats, 'r') as f:
    df = pd.read_hdf(f['asim']['block5_values'])
"""