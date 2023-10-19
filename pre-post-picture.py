import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from sys import platform
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import re
from datetime import datetime
import sys

os.environ["CDF_LIB"] = "/uio/hume/student-u58/janeod/Downloads/cdf39_0-dist-all/cdf39_0-dist/lib"
curr_path = os.getcwd()
temp_dir = "/tempdir"
full_path = curr_path+temp_dir
cdfpath = full_path + "/temp.cdf"
save_path = "/scratch/feats_FtS/extracted_feats"

from spacepy import pycdf

"""
file = open('FtS/urls/urls_fsim.txt', 'r')
urls = file.readline().split()
file.close()
link = urls[800]

loc = "fsim"
key_img = f"thg_asf_{loc}"

r = requests.get(link)
a = open('temp.cdf', 'wb')
a.write(r.content)
a.close()

df = pd.DataFrame(columns=['features', 'timestamp', 'loc'])

cdf = pycdf.CDF("temp.cdf")
num_imgs = cdf[key_img].shape[0]

img_array = cdf[key_img][100]
print(cdf)

np.save('temp.npy', img_array)

os.remove("temp.cdf")
"""

img_array = np.load('temp.npy')
"""
crop_percent = 12
num_pixels = int(img_array.shape[0]*(crop_percent/100))
temp_img = Image.fromarray(img_array)
(left, upper, right, lower) = (num_pixels, num_pixels, img_array.shape[0]-num_pixels, img_array.shape[0]-num_pixels)
temp_img2 = temp_img.crop((left, upper, right, lower))
img_array = np.asarray(temp_img2)

# Rescale to range [0,1]
# First subtract 1st percentile
percentile1 = np.percentile(img_array, 1)
img_array = img_array - percentile1

# Then divide by 99th percentile
percentile99 = np.percentile(img_array, 99)
img_array = img_array/percentile99

# Then set everything under 0 to 0, and everything over 1 to 1
img_array[img_array > 1] = 1
img_array[img_array < 0] = 0

# Make array into PIL-image
img = Image.fromarray(img_array)
img = img.resize((224,224))
#img = img.convert("RGB")
img.show()
"""

print(img_array.dtype)

imagevar = Image.fromarray(img_array, mode='I;16')
#imagevar = imagevar.convert("P")
print(img_array)
print(img_array.shape)
print(img_array.max())
print(img_array.min())
imagevar.show()