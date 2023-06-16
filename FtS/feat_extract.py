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

import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

if platform == 'win32':
    os.environ['CDF_LIB'] = "C:\\Users\janeg\Desktop\CDF\\bin"
    curr_path = os.getcwd()
    temp_dir = "\\tempdir"
    full_path = curr_path+temp_dir
    cdfpath = full_path + "\\temp.cdf"
else:
    os.environ["CDF_LIB"] = "/usr/bin/"#"/uio/hume/student-u58/janeod/.conda/envs/SVM/lib/python3.10/site-packages/spacepy"
    cdfpath = ...

from spacepy import pycdf

def retrive_urls():
    """
    locs = ["gill", "atha", "chbg", "ekat", "fsim", "fsmi", 
            "fvkn", "gako", "gbay", "gill", "inuv", "kapu",
            "kian", "kuuj", "mcgr", "nrsq", "pgeo", "rank",
            "snap", "talo", "tpas", "whit", "yknf"]
    """
    locs = ["fsim"]
    months = ["10", "11", "12", "01", "02"]
    
    os.chdir('urls')
    for loc in tqdm(locs):
        urls = []
        for month in tqdm(months):
            if month == "01" or month == "02":
                year = "2013"
            else:
                year = "2012"

            cdf_url = f"http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/{loc}/{year}/{month}/"
            r = requests.get(cdf_url)
            soup = BeautifulSoup(r.text, 'html.parser')

            for link in soup.find_all('a'):
                if re.findall(".*_asf_.*", link.get('href')):
                    urls.append(f"{cdf_url}{link.get('href')} ")
        file = open(f'urls_{loc}.txt', 'w')
        file.writelines(urls)
        file.close()
    os.chdir('..')

#retrive_urls()

file = open('urls\\urls_fsim.txt', 'r')
urls = file.readline().split()
file.close()

urls2 = [urls[0], urls[1]]

if not os.path.exists(full_path):
    os.mkdir(full_path)

os.chdir(full_path)
loc = "fsim"
key_img = f"thg_asf_{loc}"

# Define the model to be used and all it's components
model = shufflenet_v2_x1_0(weights='DEFAULT')
preproc = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.transforms()
num_feats = 1000
return_nodes = 'fc'
model_feat_extractor = create_feature_extractor(model, return_nodes=[return_nodes])

for url in tqdm(urls2):
    # Download cdf-file
    r = requests.get(url)
    a = open(cdfpath, 'wb')
    a.write(r.content)
    a.close()
    
    # Process cdf-file (get features)
    cdf = pycdf.CDF("temp.cdf")
    print(cdf)
    num_imgs = cdf[key_img].shape[0]
    feats = np.zeros([num_imgs,num_feats], dtype=np.float32)
    for i in tqdm(range(int(num_imgs))):
        img_array = cdf[key_img][i]

        # Crop image
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
        img = Image.fromarray(img_array*255)
        img = img.resize((224,224))
        img = img.convert("RGB")

        # Preprocess image to fit model, and squeeze it through!
        # Save the features aquired in an array
        image_tensor = preproc(img)
        image_batch = image_tensor.unsqueeze(0)
        pic_feats = model_feat_extractor(image_batch)
        array_feats = pic_feats[return_nodes].detach().numpy()
        feats[i] = array_feats

    # Delete cdf-file
    cdf.close()
    os.remove("temp.cdf")

#url = urls[0]

os.chdir(full_path)

"""
Do stuff
"""

# Download cdf-file
"""
r = requests.get(url)
a = open("temp.cdf", 'wb')
a.write(r.content)
a.close()
"""

os.chdir('..')
#os.rmdir(full_path)

# Save pandas as feather