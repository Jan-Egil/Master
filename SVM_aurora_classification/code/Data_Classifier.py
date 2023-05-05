import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.linear_model import Ridge

import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, inception_v3
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class Feature_Extractor:
    def __init__(self,
                 directory,
                 filenames,
                 path_feats):
        self.directory = directory
        self.filenames = filenames
        self.path_feats = path_feats
    
    def extract_features(self, model_name='shufflenet_v2_x1_0'):
        def loop(i):
            img = self.filenames[i]
            full_path = self.directory + img
            with Image.open(full_path) as im:
                image_tensor = preproc(im)
                image_batch = image_tensor.unsqueeze(0)

                pic_feats = model_feat_extractor(image_batch)
                array_feats = pic_feats['fc'].detach().numpy()
                feats[i] = array_feats
                #return array_feats

        
        if model_name == 'shufflenet_v2_x1_0':
            model = shufflenet_v2_x1_0(weights='DEFAULT')
            preproc = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])])
            num_feats = 1000
            model_feat_extractor = create_feature_extractor(model, return_nodes=['fc'])
        
        elif model_name == 'inception_v3':
            model = inception_v3(weights='DEFAULT')
            preproc = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            num_feats = 1000
            model_feat_extractor = create_feature_extractor(model, return_nodes=['fc'])
        
        else:
            raise ValueError(f"The given model ({model_name}) is not a valid option")
        
        num_pics = len(self.filenames)
        feats = np.zeros([num_pics,num_feats], dtype=np.float32)

        print(f"\nStarting feature extraction for model: {model_name}")
        Parallel(n_jobs=3, backend='sequential', require='sharedmem')(delayed(loop)(i) for i in tqdm(range(len(self.filenames))))
        """
        for i, img in enumerate(tqdm(self.filenames)):
            full_path = self.directory + img
            with Image.open(full_path) as im:
                image_tensor = preproc(im)
                image_batch = image_tensor.unsqueeze(0)

                pic_feats = model_feat_extractor(image_batch)
                array_feats = pic_feats['fc'].detach().numpy()
                feats[i] = array_feats
        """

        hf = h5py.File(self.path_feats, 'w')
        hf.create_dataset('features', data=feats)
        hf.close()

class Data_Classifier:
    def __init__(self,
                 input,
                 output,
                 test_size = 0.2,
                 average='micro'):
        self.test_size = test_size
        self.average = average
        (self.X_train, 
        self.X_test, 
        self. Y_train, 
        self.Y_test) = train_test_split(input, output, test_size=self.test_size)

    def scale(self):
        """
        Scale the data using the SKLearn Standard Scaler
        
        Input:
            - None
        Output:
            - None
        """

        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def decompose(self,
                  algo = 'PCA',
                  dims = 3,
                  kernel = 'linear'):
        """
        Reduce the dimensionality of the feature space
        
        Input:
            - algo (str, Default: 'PCA'): The dimensionality reduction technique to be used
            - dims (int, Default: 3): The number of dimensions we want to reduce to
        Output:
            - None
        """

        if algo == 'PCA':
            reducer = PCA(n_components=dims)
        elif algo == 'KernelPCA':
            reducer = KernelPCA(n_components=dims, kernel=kernel)
        else:
            raise ValueError(f"The given algorithm ({algo}) does not exist")
        
        self.X_train = reducer.fit_transform(self.X_train)
        self.X_test = reducer.transform(self.X_test)
    
    def classify(self,
                 classifier = 'SVM',
                 deg = 3,
                 C = 1,
                 gamma='scale'):
        """
        Classify the test-data using a classifier
        
        Input:
            - Classifier (str, Default: 'SVM'): The classifier to use when predicting data {'SVM', 'SVM_rbf', 'SVM_linear', 'SVM_poly', 'poly', 'LDA'}
            - deg (int, default: 3): The degree of polynomial when using the 'poly' SVC-kernel
            - gamma ({'scale', 'auto'} or float): The kernel coefficient for rbf, poly and sigmoid
        Output:
            - self.pred (Array-like): Predicted values of data
            - self.Y_test (Array-like): Actual values of data
        """

        if classifier == 'SVM' or classifier == 'SVM_rbf':
            self.clf = svm.SVC()
        elif classifier == 'SVM_linear':
            self.clf = svm.SVC(kernel='linear')
        elif classifier == 'SVM_poly':
            self.clf = svm.SVC(kernel='poly', degree=deg)
        elif classifier == 'LDA':
            self.clf = LDA()
        else:
            raise ValueError(f"The given classifier ({classifier}) does not exist")
        
        self.clf.fit(self.X_train, self.Y_train)
        self.pred = self.clf.predict(self.X_test)

        return self.pred, self.Y_test

    def accuracy(self):
        """
        Get the accuracy of the model
        
        Input:
            - None
        Output:
            - prcnt (float): Percentage of predicted values being correct
        """

        amt_corr = np.equal(self.Y_test, self.pred).sum()
        prcnt = 100*amt_corr/self.Y_test.shape[0]
        return prcnt
    
    def precision(self):
        prec = precision_score(self.Y_test, self.pred, average = self.average)
        return prec
    
    def recall(self):
        rec = recall_score(self.Y_test, self.pred, average = self.average)
        return rec
    
    def F1(self):
        score = f1_score(self.Y_test, self.pred, average = self.average)
        return score

    def plot_matrix_2(self):
        """
        Plot the confusion matrix of the results. This plot is neater
        
        Input:
            - None
        Output:
            - None (but a cute plot pops up)
        """
        
        mat = confusion_matrix(self.Y_test, self.pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=self.clf.classes_)
        disp.plot()
        plt.show()

    def plot_matrix(self):
        """
        Plot the confusion matrix of the results. Also prints it to console
        
        Input:
            - None
        Output:
            - None (but a plot pops up!)
        """

        sns.set_theme()

        mat = confusion_matrix(self.Y_test, self.pred)
        print(mat)

        fig, ax = plt.subplots()
        max_dim = mat.shape[0]
        ax.matshow(mat, cmap='OrRd')
        for i in range(max_dim):
            for j in range(max_dim):
                prcnt = mat[i,j]
                ax.text(i, j, str(prcnt), va='center', ha='center')

        plt.show()



if __name__ == "__main__":
    ...