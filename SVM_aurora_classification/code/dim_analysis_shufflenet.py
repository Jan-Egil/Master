import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from tqdm import tqdm
import os

from Data_Classifier import Data_Classifier, Feature_Extractor

os.chdir('../..')
base_dir = os.getcwd()
data_dir = base_dir + '/data'
pic_dir = '/scratch/SOPP/processed_imgs'

for picnum in range(1,5825):
    filename = str(picnum+1).zfill(5) + ".png"
    print(filename)