import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from tqdm import tqdm
import os
from sys import platform

from Data_Classifier import Data_Classifier, Feature_Extractor

if platform == 'win32':
    os.chdir('../..')
    base_dir = os.getcwd()
    data_dir = base_dir + '\data\SVM_aurora_classification'
    classification_path = data_dir + '\classifications.csv'
    os.chdir('../../..')
    base_dir_2 = os.getcwd()
    pic_dir = "C:\\Users\janeg\Desktop\oath_v1.1\images\cropped_scaled\\"
    feat_path = data_dir + '\shufflenet\\auroral_feat.h5'
else:
    os.chdir('../..')
    base_dir = os.getcwd()
    data_dir = base_dir + "/data/SVM_aurora_classification"
    classification_path = data_dir + "/classifications.csv"
    pic_dir = '/scratch/oath_v1.1/images/cropped_scaled_rotated/'
    feat_path = data_dir + '/shufflenet/auroral_feat.h5'


filename_list = []

for picnum in range(1,5825):
    filename = str(picnum).zfill(5) + ".png"
    filename_list.append(filename)

feature_extraction = Feature_Extractor(pic_dir, filename_list, feat_path)
feature_extraction.extract_features(model_name='shufflenet_v2_x1_0')


# Data Classification part

classifier = 'Ridge'
iters = 10

with h5py.File(feat_path, 'r') as f:
    features = f['features'][:]

df = pd.read_csv(classification_path, header=16)
aurora_binary = np.array(df['class2'])
aurora_6class = np.array(df['class6'])

# Looking at differences in test size

test_size_array = np.linspace(0.1, 0.5, 20)
prcnt2_array = np.zeros_like(test_size_array)
prcnt6_array = np.zeros_like(test_size_array)

for i, test_size in enumerate(tqdm(test_size_array)):
    prcnt2 = 0
    prcnt6 = 0
    for j in range(iters):
        DC2 = Data_Classifier(features, aurora_binary)#, test_size=test_size)
        DC2.scale()
        DC2.classify(classifier=classifier, alpha=test_size)
        prcnt2 += DC2.accuracy()
        
        DC6 = Data_Classifier(features, aurora_6class)#, test_size=test_size)
        DC6.scale()
        DC6.classify(classifier=classifier, alpha=test_size)
        prcnt6 += DC6.accuracy()
    prcnt2_array[i] = prcnt2/iters
    prcnt6_array[i] = prcnt6/iters


plt.figure()
plt.title("Ridge test size versus accuracy")
plt.plot(test_size_array, prcnt2_array, label="2 class Ridge")
plt.plot(test_size_array, prcnt6_array, label="6 class Ridge")
plt.xlabel("Test size")
plt.ylabel("Percent Accuracy")
plt.legend()
plt.grid()
print(f"Test size of 0.1 gives {prcnt2_array[0]}% 2class and {prcnt6_array[0]}% 6class accuracy")

# Looking at differences in (insert thing here..)

...

plt.show()