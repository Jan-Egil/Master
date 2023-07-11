import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, argv
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def define_paths():
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

        del df2
        del array_feats_first
        del loc
        del timestamps

        cdfpath = "N/A"
        save_path = "N/A"
        save_path_reduced = "reduced_feats.h5"
        save_path_binned = "binned_feats.h5"
        substorm_csv_path = "substorms.csv"


        print("Placement done!\n")
    else:
        # 0th: Define the paths
        curr_path = os.getcwd()
        temp_dir = "/tempdir"
        full_path = curr_path+temp_dir
        cdfpath = full_path + "/temp.cdf"
        save_path = "/scratch/feats_FtS/extracted_feats/"
        save_path_reduced = "/scratch/feats_FtS/reduced_feats/reduced_feats.h5"
        save_path_binned = "/scratch/feats_FtS/binned_feats/binned_feats.h5"
        substorm_csv_path = "/scratch/substorms/substorm_list.csv"

        # 1st: Gather all features in one large array-like structure
        # Tip: first make one big pandas dataframe, then turn that into array
        print("Extracting the data from file")
        df = pd.read_hdf(save_path+f"iter_0_loc_fsim.h5", key='features')

        for iter in tqdm(range(1,1686)):
            df2 = pd.read_hdf(save_path+f"iter_{iter}_loc_fsim.h5", key='features')
            df = pd.concat([df, df2], ignore_index=True)
            df.reset_index()
        del df2
        print("Data extraction done!\n")
    
    return df, cdfpath, save_path, save_path_reduced, save_path_binned, substorm_csv_path
    

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

# 2nd: Apply scaling and PCA to reduce said dimensions. (35 dims is a nice start)
# HEADS UP! One should maybe train-test-split first before scaling and reducing to be statistically correct.
# This is sort of cheating. Should look closer into this.

def scale_and_reduce(array_feats):
    print("Scaling the data..")
    sc = StandardScaler()
    array_feats = sc.fit_transform(array_feats)
    print("Scaling done!\n")

    print("Reducing dimensionality of data..")
    reducer = PCA(n_components=35)
    array_feats = reducer.fit_transform(array_feats)
    print("Reduction done!\n")

    return array_feats 

# 3rd: Connect the right features with the right locatons and timestamps in a new dataframe

def connect_reduced_with_timestamps_loc(array_feats, timestamps, loc):
    print("Placing reduced features, timestamps and location in new dataframe..")
    dict = {"feat_reduced": list(array_feats),
            "timestamp": timestamps,
            "loc": loc}
    new_df = pd.DataFrame(dict, dtype=object)
    print("Finished placing features in new dataframe!\n")
    return new_df

# 4th: Save dataframe to file. Delete variable
# Tip: Try to plot features against timestamps. See if you find some structure in the madness
def save_reduced_to_file(new_df, save_path_reduced):
    print("Placing dataframe in hdf-file..")
    new_df.to_hdf(save_path_reduced, key=f'reduced_feats', mode='w')
    print("Finished placing dataframe in hdf-file!\n")

# 5th: Extract dataframe from file. Also sort it in date-order

def extract_reduced_from_file(save_path_reduced):
    print("Extracting reduced features from file..")

    df = pd.read_hdf(save_path_reduced, key=f'reduced_feats')
    df.sort_values(by='timestamp', inplace=True, ignore_index=True)
    df.reset_index()
    
    print("File extraction complete!\n")
    return df

# 6th: Combine the features into bins of 1 minutes each. (Average of each feature on their own)

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

# 6.5th: Fill in the blank minutes that don't exist in the dataframe maybe?

"""
datetime_start = datetime(year=2012,
                          month=10,
                          day=1,
                          hour=0,
                          minute=0,
                          second=0)
datetime_stop = datetime(year=2013,
                         month=3,
                         day=1,
                         hour=0,
                         minute=0,
                         second=0)

no_data_feats = [0 for i in range(len(new_df['averaged_feats'][0]))]

datetimerange_mins = (datetime_stop-datetime_start).total_seconds()/60
print(datetimerange_mins)

empty_list = [0 for i in range(len(new_df['averaged_feats'][0]))]
print(empty_list)

print(new_df)

for minutecounter in tqdm(range(int(datetimerange_mins))):
    deltatime = timedelta(minutes=minutecounter)
    temp_datetime = datetime_start + deltatime
    if not temp_datetime in new_df['timestamp'].unique():
        dict = {'averaged_feats': empty_list,
                'timestamp': temp_datetime,
                'loc': new_df['loc'][0]}
        new_df2 = pd.DataFrame(dict)
        new_df = pd.concat([new_df, new_df2], ignore_index=True)
        new_df.reset_index()

new_df.sort_values(['timestamp'], axis=0, ignore_index=True, inplace=True)
print(new_df[0:10])
"""

# 7th: Save binned data to file
def save_binned_to_file(new_df, save_path_binned):
    print("Saving binned features to file..")
    new_df.to_hdf(save_path_binned, key=f"binned_feats", mode='w')
    print("Done saving binned to file!\n")

# 7th: Use location and timestamp-data together with onset-data to determine whether or not there has been an onset.

# Read substorm files
def substorm_extract_and_filter(df, substorm_csv_path):
    substorm_df = pd.read_csv(substorm_csv_path)

    # Filter out the substorms far away from source
    lat_fsim = 61.76
    lon_fsim = 238.77

    min_lat_fsim = lat_fsim-10
    max_lat_fsim = lat_fsim+10
    min_lon_fsim = lon_fsim-10
    max_lon_fsim = lon_fsim+10

    print(substorm_df.columns)



    droplist = []
    for i in tqdm(range(len(substorm_df.index))):
        substorm_lat = substorm_df['GLAT'][i]
        substorm_lon = substorm_df['GLON'][i]

        if min_lat_fsim <= substorm_lat <= max_lat_fsim and min_lon_fsim <= substorm_lon <= max_lon_fsim:
            print(f"{substorm_lat} - {substorm_lon}")
        else:
            droplist.append(i)

    substorm_df.drop(labels=droplist, axis=0, inplace=True)
    substorm_df.reset_index(inplace=True)
    substorm_df.drop(labels=['index', 'MLT', 'MLAT', 'GLON', 'GLAT'], axis=1, inplace=True)
    print(substorm_df)

# 8th: Create dataframe with timestamp (in 1 minute iters), whether or not there will be an onset in the next 15 mins, and the reduced features in bins


# 9th: Save dataframe to file. Delete variabele.


# 10th: 


if __name__ == "__main__":
    if len(argv) > 1:
        if argv[1] == 'test':
            platform = 'win32'
    
    # Step 1: Take features, reduce features, save to file
    df, cdfpath, save_path, save_path_reduced, save_path_binned, substorm_csv_path = define_paths()

    array_feats, timestamps, loc = separate_dataframe_contents(df)

    array_feats = scale_and_reduce(array_feats)

    new_df = connect_reduced_with_timestamps_loc(array_feats, timestamps, loc)

    del array_feats, timestamps, loc

    save_reduced_to_file(new_df, save_path_reduced)

    del new_df, df

    # Step 2: Take reduced features, bin together, save to file.

    df = extract_reduced_from_file(save_path_reduced)

    new_df = feature_binning(df)

    save_binned_to_file(new_df, save_path_binned)

    del df, new_df