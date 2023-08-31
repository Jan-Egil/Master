import numpy as np

arrays = ['alphas',
          'balacc', 'balacc_std', 
          'falsepos', 'falsepos_std', 
          'recall', 'recall_std']
suffix = ".npy"

path_for_2 = "hyperparam_35feat_1min/"
    
for array in arrays:
    file1_path = array + suffix
    file2_path = path_for_2 + file1_path
    savepath = './big_' + path_for_2 + file1_path
    array1 = np.load(file1_path)
    array2 = np.load(file2_path)
    
    total_len = array1.shape[0] + array2.shape[0]
    longarray = np.concatenate([array1, array2])
    print(longarray)
    np.save(savepath, longarray)

print("Jobs done!")