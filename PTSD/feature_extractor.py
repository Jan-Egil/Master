import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class feature_extractor:
    def __init__(self, picture_path, feature_path, filenames):
        self.img_path = picture_path
        self.feature_path = feature_path
        self.filenames = filenames
        self.num_imgs = len(filenames)
    
    def preproc_shufflenet_v2_x1_0(self):
        self.model = shufflenet_v2_x1_0(weights='DEFAULT')
        self.preproc = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.transforms()
        self.num_feats = 1000
        self.return_node = 'fc'
        self.model_name = 'Shufflenet_v2_x1_0'


    def preproc_inception_v3(self):
        self.model = inception_v3(weights='DEFAULT')
        self.preproc = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        self.num_feats = 1000
        self.return_node = 'fc'
        self.model_name = 'Inception_v3'

    def extract_features(self):
        feature_extractor = create_feature_extractor(model=self.model, return_nodes=[self.return_node])
        features = np.zeros([self.num_imgs,self.num_feats], dtype=np.float32)

        print(f"\nStarting feature extraction for model: {self.model_name}")
        for i, img_filename in enumerate(tqdm(self.filenames)):
            full_path = self.img_path + img_filename
            with Image.open(full_path) as im:
                image_tensor = self.preproc(im)
                image_batch = image_tensor.unsqueeze(0)

                pic_feats = feature_extractor(image_batch)
                array_feats = pic_feats[self.return_node].detach().numpy()
                features[i] = array_feats
        