import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, argv
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def define_paths():
    if platform == 'win32':
        cdfpath = "N/A"
        save_path = "N/A"
        save_path_reduced = "clustered_feats.h5"
        save_path_binned = "binned_feats.h5"
        substorm_csv_path = "substorms.csv"
        master_df_path = "master_fsim.h5"
        master_df_trainable_path = "master_trainable_fsim.h5"
    else:
        curr_path = os.getcwd()
        temp_dir = "/tempdir"
        full_path = curr_path+temp_dir
        cdfpath = full_path + "/temp.cdf"
        save_path = "/scratch/feats_FtS/extracted_feats/"
        save_path_reduced = "/scratch/feats_CLtS/reduced_feats/clustered_feats.h5"
        save_path_binned = "/scratch/feats_CLtS/binned_feats/binned_feats.h5"
        substorm_csv_path = "/scratch/substorms/substorms_forsyth.csv"
        master_df_path = "/scratch/feats_CLtS/master_df/master_fsim.h5"
        master_df_trainable_path = "/scratch/feats_CLtS/master_df/master_trainable_fsim.h5"
    return cdfpath, save_path, save_path_reduced, save_path_binned, substorm_csv_path, master_df_path, master_df_trainable_path

def fetch_initial_data(save_path):
    if platform == 'win32':
        array_feats_first = np.random.random((100,1000))
        loc = ['fsim' for i in range(100)]
        timestamps = [datetime.now()+timedelta(seconds=20*i) for i in range(100)]

        dict = {"features": [array_feats_first[0]],
                "timestamp": timestamps[0],
                "loc": loc[0]}
        df = pd.DataFrame(dict, dtype=object)
        print("Placing randomly generated features in dataframe..")
        for i in tqdm(range(1,100)):
            dict = {"features": [array_feats_first[i]],
                    "timestamp": timestamps[i],
                    "loc": loc[i]}
            df2 = pd.DataFrame(dict)
            df = pd.concat([df, df2], ignore_index=True)
            df.reset_index()
        print("Placement done!")
    else:
        print("Extracting the data from file")
        df = pd.read_hdf(save_path+f"iter_0_loc_fsim.h5", key='features')
        
        for iter in tqdm(range(1,1686)):
            df2 = pd.read_hdf(save_path+f"iter_{iter}_loc_fsim.h5", key='features')
            df = pd.concat([df, df2], ignore_index=True)
            df.reset_index()
        del df2
        print("Data extraction done!\n")
    return df

def separate_dataframe_contents(df):
# Define and extract features-column into its own array
    num_points = len(df.index)
    num_feats = 1000
    array_feats = np.zeros((num_points, num_feats))
    timestamps = []
    loc = []

    print("Extracting data from dataframe into separate arrays..")
    for i in tqdm(range(num_points)):
        array_feats[i] = df['features'][i]
        timestamps.append(df['timestamp'][i])
        loc.append(df['loc'][i])
    print("Extraction from dataframe done!\n")
    return array_feats, timestamps, loc

def scale_and_cluster(array_feats):
    print("Scaling the data..")
    sc = StandardScaler()
    array_feats = sc.fit_transform(array_feats)
    print("Scaling done!\n")

    print("clustering and labeling each picture..")
    n_reduced = 6
    reducer = KMeans(n_clusters=n_reduced, n_init='auto')
    array_feats = reducer.fit_transform(array_feats)
    print("Reduction done!\n")

    return array_feats 

def normalize_vectors(array_feats):
    num_pics = ...

def connect_reduced_with_timestamps_loc(array_feats, timestamps, loc):
    print("Placing labeled features, timestamps and location in new dataframe..")
    dict = {"feat_reduced": list(array_feats),
            "timestamp": timestamps,
            "loc": loc}
    new_df = pd.DataFrame(dict, dtype=object)
    print("Finished placing features in new dataframe!\n")
    return new_df

def save_reduced_to_file(new_df, save_path_reduced):
    print("Placing dataframe in hdf-file..")
    new_df.to_hdf(save_path_reduced, key=f'reduced_feats', mode='w')
    print("Finished placing dataframe in hdf-file!\n")

if __name__ == "__main__":
    if len(argv) > 1:
        if argv[1] == 'test':
            platform = 'win32'

    # Step 0: Define paths
    paths = define_paths()
    cdfpath = paths[0]
    save_path = paths[1]
    save_path_reduced = paths[2]
    save_path_binned = paths[3]
    substorm_csv_path = paths[4]
    master_df_path = paths[5]
    master_df_trainable_path = paths[6]

    # Step 1: Take features, cluster and label them, save to file
    y_or_n = input("Do you want to cluster the features? [Y/n] ")

    if y_or_n == "Y" or y_or_n == "y":
        df = fetch_initial_data(save_path)

        array_feats, timestamps, loc = separate_dataframe_contents(df)

        array_feats = scale_and_cluster(array_feats)

        new_df = connect_reduced_with_timestamps_loc(array_feats, timestamps, loc)

        del array_feats, timestamps, loc

        save_reduced_to_file(new_df, save_path_reduced)

        del new_df, df
    
    # Step 2: Take image labels, bin together, save to file
    y_or_n = input("Do you want to bin the image labels? [Y/n] ")

