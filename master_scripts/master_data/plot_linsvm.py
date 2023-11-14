import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

num_feats_list = [35,6,4]
minbins_list = [1,5]
dataset_list = ['FtS', 'CLtS', 'ULtS']

print("")
plt.figure()
for dataset in dataset_list:
    for minbins in minbins_list:
        for num_feats in num_feats_list:
            filepath = f"LinSVM_{dataset}_{minbins}bins_{num_feats}feats.csv"
            df = pd.read_csv(filepath)
            Cs = np.array(df.Cs)
            balaccs = np.array(df.balanced_accuracy)
            balaccs_std = np.array(df.balanced_accuracy_std)
            recalls = np.array(df.recall)
            recalls_std = np.array(df.recall_std)
            fpr = np.array(df.fpr)
            fpr_std = np.array(df.fpr_std)
            fitting_time = np.array(df.fitting_time)
            fitting_time_std = np.array(df.fitting_time_std)
            classification_time = np.array(df.classification_time)
            classification_time_std = np.array(df.classification_time_std)
            total_time = np.array(df.total_time)
            total_time_std = np.array(df.total_time_std)

            best_balacc_idx = np.argmax(balaccs)

            best_balacc = balaccs[best_balacc_idx]
            best_balacc_std = balaccs_std[best_balacc_idx]

            best_recall = recalls[best_balacc_idx]
            best_recall_std = recalls_std[best_balacc_idx]

            best_fpr = fpr[best_balacc_idx]
            best_fpr_std = fpr_std[best_balacc_idx]

            best_fitting_time = fitting_time[best_balacc_idx]
            best_fitting_time_std = fitting_time_std[best_balacc_idx]

            best_classification_time = classification_time[best_balacc_idx]
            best_classification_time_std = classification_time_std[best_balacc_idx]

            best_total_time = best_classification_time + best_fitting_time
            best_total_time_std = total_time[best_balacc_idx]

            best_C = Cs[best_balacc_idx]

            print(f"{dataset} {minbins}bins {num_feats} feats. C = {best_C:e}")
            print(f"Balanced accuracy | {best_balacc:%} ± {best_balacc_std:%}")
            print(f"Recall            | {best_recall:%} ± {best_recall_std:%}")
            print(f"FPR               | {best_fpr:%} ± {best_fpr_std:%}")
            print(f"Fit time          | {best_fitting_time:e} ± {best_fitting_time_std:e}")
            print(f"Classify time     | {best_classification_time:e} ± {best_classification_time_std:e}")
            print(f"Total time        | {best_total_time:e} ± {best_total_time_std:e}")
            print("")

            if num_feats == 35:
                plt.plot(Cs ,balaccs, label=f"{dataset} | {minbins}minbin")
plt.xscale('log')
plt.ylabel("Balanced accuracy score", fontsize='large')
plt.xlabel("Hyperparameter C", fontsize='large')
plt.title("Balanced Accuracy against hyperparameter C\nReduced to 35 features")
plt.legend()
plt.show()