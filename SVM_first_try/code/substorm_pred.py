import numpy as np
import matplotlib.pyplot as plt
import h5py

from Data_Classifier import Data_Classifier


yearmonth = ['201011', '201012', '201101', '201102']
dims = 30

path_feats1 = '/scratch/oath_v1.1/features/auroral_feat.h5'
path_classification = '/scratch/oath_v1.1/classifications/classifications.csv'
path_substorms = '/scratch/substorms/substorm_list.csv'
path_feats2 = '/scratch/asim/' + yearmonth[0] + ".hdf"

"""
Train classifier
"""


"""
with h5py.File(path_feats1, 'r') as f:
    feats_training = f['Logits'][:]
"""

"""
Use classifier to create df with {datetime, class6-aurora}
"""

with h5py.File(path_feats2, 'r') as f:
    asim = f['asim']
    feats = asim['block1_values'][:]
    print(feats.shape)
    print(asim['block4_values'][:])
    print(asim.keys())


"""
Train new classifier for substorm onset prediction
"""


"""
Test classifier for onset prediction
"""