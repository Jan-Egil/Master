import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm


class Data_Classifier:
    def __init__(self,
                 input,
                 output,
                 test_size = 0.2):
        (self.X_train, 
        self.X_test, 
        self. Y_train, 
        self.Y_test) = train_test_split(input, output, test_size=test_size)

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
                  dims = 3):
        """
        Reduce the dimensionality
        Input:
            - algo (str, Default: 'PCA'): The dimensionality reduction technique to be used
            - dims (int, Default: 3): The number of dimensions we want to reduce to
        Output:
            - None
        """

        if algo == 'PCA':
            reducer = PCA(n_components=dims)
        else:
            raise ValueError(f"The given algorithm ({algo}) does not exist")
        
        self.X_train = reducer.fit_transform(self.X_train)
        self.X_test = reducer.transform(self.X_test)
    
    def classify(self,
                 classifier = 'SVM',
                 deg = 3):
        """
        Classify the test-data using a classifier
        
        Input:
            - Classifier (str, Default: 'SVM'): The classifier to use when predicting data {'SVM', 'SVM_rbf', 'SVM_linear', 'SVM_poly', 'poly', 'LDA'}
            - deg (int, default: 3): The degree of polynomial when using the 'poly' SVC-kernel
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
    path_feats = '/scratch/oath_v1.1/features/auroral_feat.h5'
    path_classification = '/scratch/oath_v1.1/classifications/classifications.csv'

    with h5py.File(path_feats, 'r') as f:
        feats = f['Logits'][:]
    
    df = pd.read_csv(path_classification, header=16)
    classified_aurora = np.array(df['class6'])

    DC = Data_Classifier(feats, classified_aurora)
    DC.scale()
    DC.decompose(algo='PCA', dims=10)
    DC.classify(classifier='LDA')
    DC.plot_matrix()