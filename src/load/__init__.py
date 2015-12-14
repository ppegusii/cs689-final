from __future__ import print_function
import gzip
import json
import pandas as pd


# Given a data file, such as
# 'data/kasteren/2010/datasets/houseA/resampled_data.csv.gz'
# this function returns a pandas dataframe of size N by M+1.
# N is the number of timesteps.
# M is the number of sensors.
# The last column is the activity label.
def data(fileName, dtype_str=False):

    with gzip.open(fileName, 'rb') as f:
        if dtype_str:
            df = pd.read_csv(
                f,
                parse_dates=[0],
                index_col=0,
                dtype=object
             )
        else:
            df = pd.read_csv(
                f,
                parse_dates=[0],
                index_col=0
             )

    return df


# Given a name file, such as
# 'data/kasteren/2010/datasets/houseA/names.json'
# this function returns the following map of maps:
# {
#   "activities": { activity_id: "activity name"},
#   "sensors": { sensor_id: "sensor name"},
# }
def names(fileName):
    with open(fileName, 'rb') as f:
        m = json.load(f)

    def stringKeysToInts(d):
        return {int(i[0]): i[1] for i in d.items()}
    return {i[0]: stringKeysToInts(i[1]) for i in m.items()}
