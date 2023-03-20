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
        sns.set_theme()
        sns.set_style('white')

    def scale(self):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def decompose(self,
                  algo = 'PCA',
                  dims = 3):
        if algo == 'PCA':
            reducer = PCA(n_components=dims)
        else:
            raise ValueError(f"The given algorithm ({algo}) does not exist")
        
        self.X_train = reducer.fit_transform(self.X_train)
        self.X_test = reducer.transform(self.X_test)
    
    def classify(self,
                 classifier = 'SVM',
                 deg = 3):
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

        return self.pred
    
    def plot_matrix(self):
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


"""
sc = StandardScaler()
path_feats = '/scratch/oath_v1.1/features/auroral_feat.h5'
path_classification = '/scratch/oath_v1.1/classifications/classifications.csv'

with h5py.File(path_feats, 'r') as f:
    feats = f['Logits'][:]

df = pd.read_csv(path_classification, header=16)
aurora_binary = np.array(df['class2'])
aurora_type = np.array(df['class6'])
"""
"""
num_components = np.arange(2,30)
#num_components = np.array([20])
prcnt_array = np.zeros(num_components.shape[0])

iters = 5

for i, dims in enumerate(tqdm(num_components)):
    sum = 0
    for j in range(iters):
        reducer = PCA(n_components = dims)
        
        feats_train, feats_test, abin_train, abin_test = train_test_split(feats, aurora_binary, test_size = 0.2)
        #feats_train, feats_test, abin_train, abin_test = train_test_split(feats, aurora_type, test_size=0.2)
        feats_train = sc.fit_transform(feats_train)
        feats_test = sc.transform(feats_test)

        feats_train = reducer.fit_transform(feats_train)
        feats_test = reducer.transform(feats_test)

        clf = svm.SVC()
        clf.fit(feats_train, abin_train)

        pred = clf.predict(feats_test)
        mat = confusion_matrix(abin_test, pred)
        
"""
"""
        amt_corr = 0
        for k in range(3):
            for l in range(3):
                amt_corr += mat[k][l]
                amt_corr += mat[k+3][l+3]
        sum += amt_corr/abin_test.shape[0]
"""
"""

        amt_corr = np.equal(abin_test, pred).sum()
        sum += amt_corr/abin_test.shape[0]
    sum /= iters
    prcnt_array[i] = sum


plt.plot(num_components, prcnt_array)
plt.grid()
plt.xlabel("Num dims")
plt.ylabel("Percent")
plt.show()
"""
"""
iters = 100

prcnt_array = np.zeros(iters)

for i in tqdm(range(iters)):
    #feats_train, feats_test, abin_train, abin_test = train_test_split(feats, aurora_binary, test_size=0.2)
    feats_train, feats_test, abin_train, abin_test = train_test_split(feats, aurora_type, test_size=0.2)    

    clf = LDA()
    clf.fit(feats_train, abin_train)

    pred = clf.predict(feats_test)
    mat = confusion_matrix(abin_test, pred)
    amt_corr = np.equal(abin_test, pred).sum()
    prcnt = amt_corr/abin_test.shape[0]
    prcnt_array[i] = prcnt
    #print(prcnt)
    #print(mat)

print(np.mean(prcnt_array))
print(np.std(prcnt_array))

#for i in range(abin_test.shape[0]):
#    print(abin_test[i] == pred[i])

#print(np.equal(abin_test, pred))
"""
"""
xs = feats[:,0]#[:100]
ys = feats[:,1]#[:100]
zs = feats[:,2]#[:100]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()
"""