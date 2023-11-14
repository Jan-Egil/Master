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
    n_reduced = 35
    reducer = KMeans(n_clusters=n_reduced, n_init='auto')
    array_feats = reducer.fit_transform(array_feats)
    print("Reduction done!\n")

    print("Normalizing the array")
    array_feats_norm = np.linalg.norm(array_feats, axis=1)
    array_feats = ((array_feats.T/array_feats_norm).T)**2
    print("Normalization done!")

    return array_feats 

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

def extract_reduced_from_file(save_path_reduced):
    print("Extracting reduced features from file..")

    df = pd.read_hdf(save_path_reduced, key=f'reduced_feats')
    df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    df.reset_index()

    print("File extraction complete!\n")
    return df

def feature_binning(df):
    print("Started binning together features in same minute..")
    new_df = pd.DataFrame(columns=['averaged_feats', 'timestamp', 'loc'])

    num_points = len(df.index)
    num_in_minute = 0
    set_minute = df['timestamp'][0].minute
    set_hour = df['timestamp'][0].hour
    num_reduced_feats = len(df['feat_reduced'][0])

    for i in tqdm(range(num_points)):
        timestamp = df['timestamp'][i]
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minute = timestamp.minute

        # If it's the last element, cut it off entirely. Fill in with previous 
        if i == num_points-1:
            relevant_df = df[['feat_reduced', 'timestamp', 'loc']][i-num_in_minute:i]
            relevant_df.reset_index(inplace=True)
            num_points_in_relevant = len(relevant_df.index)
            average_list = []
            for feature_elem in range(num_reduced_feats):
                average_val = 0
                for elem_in_df in range(num_points_in_relevant):
                    feature_value = relevant_df['feat_reduced'][elem_in_df][feature_elem]
                    average_val += feature_value
                average_val = average_val/num_points_in_relevant
                average_list.append(average_val)

            relevant_timestamp = relevant_df['timestamp'][0]
            new_timestamp = datetime(year=relevant_timestamp.year,
                                    month=relevant_timestamp.month,
                                    day=relevant_timestamp.day,
                                    hour=relevant_timestamp.hour,
                                    minute=relevant_timestamp.minute,
                                    second=0)
            dict = {'averaged_feats': [np.array(average_list)],
                    'timestamp': new_timestamp,
                    'loc': relevant_df['loc'][0]}
            new_df2 = pd.DataFrame(dict)
            new_df = pd.concat([new_df, new_df2], ignore_index=True)
            new_df.reset_index()
            num_in_minute = 1
            set_minute = minute
            set_hour = hour

        # If it's the same datetime as the one above, continue the loop
        elif minute == set_minute and hour == set_hour:
            num_in_minute += 1

        # If it's a new datetime, fill in all the previous elements and bin them together
        else:
            relevant_df = df[['feat_reduced', 'timestamp', 'loc']][i-num_in_minute:i]
            relevant_df.reset_index(inplace=True)
            num_points_in_relevant = len(relevant_df.index)
            average_list = []
            for feature_elem in range(num_reduced_feats):
                average_val = 0
                for elem_in_df in range(num_points_in_relevant):
                    feature_value = relevant_df['feat_reduced'][elem_in_df][feature_elem]
                    average_val += feature_value
                average_val = average_val/num_points_in_relevant
                average_list.append(average_val)

            relevant_timestamp = relevant_df['timestamp'][0]
            new_timestamp = datetime(year=relevant_timestamp.year,
                                    month=relevant_timestamp.month,
                                    day=relevant_timestamp.day,
                                    hour=relevant_timestamp.hour,
                                    minute=relevant_timestamp.minute,
                                    second=0)
            
            dict = {'averaged_feats': [np.array(average_list)],
                    'timestamp': new_timestamp,
                    'loc': relevant_df['loc'][0]}
            
            new_df2 = pd.DataFrame(dict)
            new_df = pd.concat([new_df, new_df2], ignore_index=True)
            new_df.reset_index()
            num_in_minute = 1
            set_minute = minute
            set_hour = hour
    print("Finished binning features together!\n")
    return new_df

def save_binned_to_file(new_df, save_path_binned):
    print("Saving binned features to file..")
    new_df.to_hdf(save_path_binned, key=f"binned_feats", mode='w')
    print("Done saving binned to file!\n")

def fetch_binned_from_file(save_path_binned):
    print("Started extraction of binned feature data..")

    df = pd.read_hdf(save_path_binned, key=f'binned_feats')
    df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    df.reset_index()

    print("Finished extracting binned feature data!\n")
    return df

def substorm_extract_and_filter(df, substorm_csv_path):
    substorm_df = pd.read_csv(substorm_csv_path)

    # Filter out the substorms far away from source
    lat_fsim = 61.76
    lon_fsim = 238.77

    min_lat_fsim = lat_fsim-15
    max_lat_fsim = lat_fsim+15
    min_lon_fsim = lon_fsim-15
    max_lon_fsim = lon_fsim+15



    droplist = []
    print("Removing substorms that are too far away...")
    for i in tqdm(range(len(substorm_df.index))):
        substorm_lat = substorm_df['GLAT'][i]
        substorm_lon = substorm_df['GLON'][i]

        if min_lat_fsim <= substorm_lat <= max_lat_fsim and min_lon_fsim <= substorm_lon <= max_lon_fsim:
            continue
        else:
            droplist.append(i)

    substorm_df.drop(labels=droplist, axis=0, inplace=True)
    substorm_df.reset_index(inplace=True)
    substorm_df.drop(labels=['index', 'MLT', 'MLAT', 'GLON', 'GLAT'], axis=1, inplace=True)
    print("Removed substorms too far away!\n")

    print("Removing substorms without coverage the last 45 minutes before event..")
    droplist = []
    
    for i in tqdm(range(len(substorm_df.index))):
        substorm_onset_time = substorm_df['Date_UTC'][i]
        year = int(substorm_onset_time[0:4])
        month = int(substorm_onset_time[5:7])
        date = int(substorm_onset_time[8:10])
        hour = int(substorm_onset_time[11:13])
        minute = int(substorm_onset_time[14:16])
        second = 0
        substorm_onset_time_datetime = datetime(year=year,
                                                month=month,
                                                day=date,
                                                hour=hour,
                                                minute=minute,
                                                second=second)
        for j in range(45):
            checktime = substorm_onset_time_datetime - timedelta(minutes=j)
            if not checktime in set(df['timestamp']):
                droplist.append(i)
                break


    substorm_df.drop(labels=droplist, axis=0, inplace=True)
    substorm_df.reset_index(inplace=True)
    substorm_df.drop(labels=['index'], axis=1, inplace=True)

    print("Removed substorms without feature data!")
    print(substorm_df)
    return substorm_df

def create_master_dataframe(df, substorm_df):
    print("Creating the final master dataframe..")

    df['substorm_onset'] = 0
    for i in tqdm(range(len(substorm_df.index))):
        substorm_onset_time = substorm_df['Date_UTC'][i]
        year = int(substorm_onset_time[0:4])
        month = int(substorm_onset_time[5:7])
        date = int(substorm_onset_time[8:10])
        hour = int(substorm_onset_time[11:13])
        minute = int(substorm_onset_time[14:16])
        second = 0
        substorm_onset_time_datetime = datetime(year=year,
                                                month=month,
                                                day=date,
                                                hour=hour,
                                                minute=minute,
                                                second=second)
        for j in range(15):
            checktime = substorm_onset_time_datetime - timedelta(minutes=j)
            idx = df.index[df['timestamp'] == checktime].tolist()[0]
            df.at[idx,'substorm_onset'] = 1
    
    print("Finished creating the final master dataframe!\n")
    print(df)
    return df

def save_master_dataframe(master_df, master_df_path):
    print("Saving master dataframe to file..")
    master_df.to_hdf(master_df_path, key=f"final_feats", mode='w')
    print("Master dataframe saved to file!\n")

def fetch_master_dataframe(master_df_path):
    print("Fetching the master dataframe from file..")

    master_df = pd.read_hdf(master_df_path, key=f'final_feats')
    master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    master_df.reset_index()

    print("Master dataframe has been fetched!\n")
    return master_df

def make_trainable_column(master_df):
    print("Making master df with trainable column..")
    master_df['trainable'] = 0

    for i in tqdm(range(len(master_df.index))):
        trigger = 0
        df_time = master_df['timestamp'][i]
        for j in range(30):
            idx = i-j
            if idx <= 0:
                trigger = 1
                continue
            checktime = df_time- timedelta(minutes=j)
            df_time_prev = master_df['timestamp'][idx]
            if not checktime == df_time_prev:
                trigger = 1
        if trigger == 0:
            master_df.at[i, 'trainable'] = 1

    print("Finished making master df with trainable column!\n")
    return master_df

def save_trainable_master_df(master_trainable_df, master_df_trainable_path):
    print("Saving master df with trainable columns to file..")
    master_trainable_df.to_hdf(master_df_trainable_path, key=f"final_feats", mode='w')
    print("Master df with trainable columns saved to file!\n")

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
    if y_or_n == "Y" or y_or_n == "y":
        df = extract_reduced_from_file(save_path_reduced)

        new_df = feature_binning(df)

        save_binned_to_file(new_df, save_path_binned)

        del df, new_df
    
    
    y_or_n = input("Do you want to create the master dataframe w/ substorms? [Y/n] ")
    if y_or_n == "Y" or y_or_n == "y":
        df = fetch_binned_from_file(save_path_binned)

        substorm_df = substorm_extract_and_filter(df, substorm_csv_path)

        master_df = create_master_dataframe(df, substorm_df)

        save_master_dataframe(master_df, master_df_path)

        del master_df
    

    y_or_n = input("Do you want to expand the master dataframe w/ trainable column? [Y/n] ")
    if y_or_n == "Y" or y_or_n == "y":
        master_df = fetch_master_dataframe(master_df_path)

        master_trainable_df = make_trainable_column(master_df)

        save_trainable_master_df(master_trainable_df, master_df_trainable_path)

