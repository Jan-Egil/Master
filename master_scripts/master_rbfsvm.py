import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, exit
from time import perf_counter

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, balanced_accuracy_score

def model_runner_rbfsvm(num_feats, minbins, dataset, Cs):
    kfold = KFold(n_splits=5, shuffle=False)

    master_df_path = f"/scratch/feats_{dataset}/master_df/master_trainable_fsim_{num_feats}feat"
    if minbins == 1:
        master_df_path += ".h5"
    elif minbins == 5:
        master_df_path += "_5min.h5"
    
    master_df = pd.read_hdf(master_df_path, key=f"final_feats")
    master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    
    num_imgs = len(master_df.index)

    array_feats = np.zeros((num_imgs, num_feats))
    substorm_onset = np.zeros(num_imgs)
    trainable = np.zeros(num_imgs)
    timestamps_list = []

    for i in range(num_imgs):
        array_feats[i] = master_df['averaged_feats'][i]
        substorm_onset[i] = master_df['substorm_onset'][i]
        trainable[i] = master_df['trainable'][i]
        timestamps_list.append(master_df['timestamp'][i])
    timestamps = np.array(timestamps_list)

    recalls = np.zeros_like(Cs)
    recalls_std = np.zeros_like(Cs)

    balaccs = np.zeros_like(Cs)
    balaccs_std = np.zeros_like(Cs)

    falsepos = np.zeros_like(Cs)
    falsepos_std = np.zeros_like(Cs)

    fitting_time = np.zeros_like(Cs)
    fitting_time_std = np.zeros_like(Cs)

    classification_time = np.zeros_like(Cs)
    classification_time_std = np.zeros_like(Cs)

    total_time = np.zeros_like(Cs)
    total_time_std = np.zeros_like(Cs)

    for Cidx, C in enumerate(tqdm(Cs)):
        recall_list = []
        balanced_acc_list = []
        fpr_list = []

        fitting_time_list = []
        predicting_time_list = []
        total_time_list = []

        model = SVC(C=C, 
                    kernel='rbf', 
                    class_weight='balanced',
                    cache_size=1000,
                    max_iter=500)

        for idxs_train, idxs_test in kfold.split(array_feats):
            train_idxs_filtered = []
            test_idxs_filtered = []
            for train_idx in idxs_train:
                if trainable[train_idx] == 1:
                    train_idxs_filtered.append(train_idx)
            for test_idx in idxs_test:
                if trainable[test_idx] == 1:
                    test_idxs_filtered.append(test_idx)
            
            num_imgs_train = len(train_idxs_filtered)
            num_imgs_test = len(test_idxs_filtered)
            
            if minbins == 1:
                X_train = np.zeros((num_imgs_train, num_feats*30))
                X_test = np.zeros((num_imgs_test, num_feats*30))
                Y_train = substorm_onset[train_idxs_filtered]
                Y_test = substorm_onset[test_idxs_filtered]

                for i, train_idx in enumerate(train_idxs_filtered):
                    X_train[i] = array_feats[train_idx-29:train_idx+1].flatten()
                for j, test_idx in enumerate(test_idxs_filtered):
                    X_test[j] = array_feats[test_idx-29:test_idx+1].flatten()
            
            elif minbins == 5:
                X_train = np.zeros((num_imgs_train, num_feats*6))
                X_test = np.zeros((num_imgs_test, num_feats*6))
                Y_train = substorm_onset[train_idxs_filtered]
                Y_test = substorm_onset[test_idxs_filtered]

                for i, train_idx in enumerate(train_idxs_filtered):
                    X_train[i] = array_feats[train_idx-5:train_idx+1].flatten()
                for j, test_idx in enumerate(test_idxs_filtered):
                    X_test[j] = array_feats[test_idx-5:test_idx+1].flatten()
            
            start_fitting = perf_counter()
            clf = model.fit(X_train, Y_train)
            stop_fitting = perf_counter()
            model_fitting_time = stop_fitting-start_fitting

            start_predicting = perf_counter()
            Y_pred = clf.predict(X_test)
            stop_predicting = perf_counter()
            model_predicting_time = stop_predicting-start_predicting
            model_total_time = model_fitting_time+model_predicting_time

            tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
            model_fpr = fp/(fp+tn)
            model_recall = recall_score(Y_test, Y_pred)
            model_balanced_accuracy = balanced_accuracy_score(Y_test, Y_pred)

            recall_list.append(model_recall)
            balanced_acc_list.append(model_balanced_accuracy)
            fpr_list.append(model_fpr)
            fitting_time_list.append(model_fitting_time)
            predicting_time_list.append(model_predicting_time)
            total_time_list.append(model_total_time)

        recalls[Cidx] = np.mean(recall_list)
        recalls_std[Cidx] = np.std(recall_list)

        balaccs[Cidx] = np.mean(balanced_acc_list)
        balaccs_std[Cidx] = np.std(balanced_acc_list)

        falsepos[Cidx] = np.mean(fpr_list)
        falsepos_std[Cidx] = np.std(fpr_list)

        fitting_time[Cidx] = np.mean(fitting_time_list)
        fitting_time_std[Cidx] = np.std(fitting_time_list)

        classification_time[Cidx] = np.mean(predicting_time_list)
        classification_time_std[Cidx] = np.std(predicting_time_list)

        total_time[Cidx] = np.mean(total_time_list)
        total_time_std[Cidx] = np.std(total_time_list)
    
    csv_path = f"RBFSVM_{dataset}_{minbins}bins_{num_feats}feats.csv"

    df_to_file = pd.DataFrame(data={'Cs': Cs,
                                    'recall': recalls,
                                    'recall_std': recalls_std,
                                    'balanced_accuracy': balaccs,
                                    'balanced_accuracy_std': balaccs_std,
                                    'fpr': falsepos,
                                    'fpr_std': falsepos_std,
                                    'fitting_time': fitting_time,
                                    'fitting_time_std': fitting_time_std,
                                    'classification_time': classification_time,
                                    'classification_time_std': classification_time_std,
                                    'total_time': total_time,
                                    'total_time_std': total_time_std})
    
    df_to_file.to_csv(csv_path)

if __name__ == "__main__":
    num_feats_list = [4,6,35]
    minbins_list = [1,5]
    dataset_list = ['FtS', 'CLtS', 'ULtS']
    Cs = np.logspace(-8,8,200)

    if not os.path.exists('master_data'):
        os.mkdir('master_data')
    os.chdir('master_data')

    for num_feats in tqdm(num_feats_list):
        for minbins in tqdm(minbins_list):
            for dataset in tqdm(dataset_list):
                model_runner_rbfsvm(num_feats, minbins, dataset, Cs)