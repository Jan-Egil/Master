import numpy as np
import matplotlib.pyplot as plt
import h5py

features_path = '/scratch/SOPP/features/features.h5'
model_path = '/scratch/SOPP/model/'

# Import images from hdf-file

with h5py.File(features_path, 'r') as f:
    features = np.array(f.get('1000_features'))

num_ims = features.shape[0]


# Train model using PCA and SVM

# Export model to file