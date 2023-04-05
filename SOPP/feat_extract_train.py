import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import h5py

import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0 as shufflenet
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

model = shufflenet(weights=None)
model.eval()
train_nodes, eval_nodes = get_graph_node_names(shufflenet())

path_img_info = '/scratch/oath_v1.1/classifications/classifications.csv'
image_path = '/scratch/SOPP/processed_imgs/'
feature_path = '/scratch/SOPP/features/features.h5'

# Throw images through model

df = pd.read_csv(path_img_info, header=16)
num = np.array(df['picNum'])

model_feat_extractor = create_feature_extractor(model, return_nodes=['fc'])

data = np.zeros([num.shape[0],1000])


for i in tqdm(range(num.shape[0])):
    filename = str(i+1).zfill(5) + ".png"
    full_path_fetch = image_path + filename
    preproc = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])])
    
    with Image.open(full_path_fetch) as im:
        image_tensor = preproc(im)
        image_batch = image_tensor.unsqueeze(0)

        feats = model_feat_extractor(image_batch)
        feats_array = feats['fc'].detach().numpy()
        data[i] = feats_array

print(data.shape)

hf = h5py.File(feature_path, 'w')
hf.create_dataset('1000_features', data=data)
hf.close()