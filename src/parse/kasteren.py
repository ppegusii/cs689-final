#!/usr/bin/env python
from __future__ import print_function
import argparse
import gzip
import json
import numpy as np
import pandas as pd
import re
import sys


def main():
    args = parseArgs(sys.argv)
    aNames = activityNames(args.activity_names)
    # print(aNames)
    sNames = sensorNames(args.sensor_names)
    # print(sNames)
    sValues = sensorValues(args.sensor_values, sNames.keys())
    # print(sValues)
    aLabels = activityLabels(args.activity_labels)
    # print(aLabels)
    data = combine(sValues, aLabels)
    # print(data)
    with gzip.open(args.data, 'w') as f:
        data.to_csv(f, float_format='%.0f')
    resampled = resample(data, args.timestep)
    with gzip.open(args.resampled_data, 'w') as f:
        resampled.to_csv(f, float_format='%.0f')
    names = {
        'activities': aNames,
        'sensors': sNames,
    }
    with open(args.names, 'w') as f:
        json.dump(names, f)


# Down-sample the pandas dataframe data to the given timestep.
# Right now, the down-sample results in the majority for that
# timestep, unless the majority is 0.
# In that case it will assign the next most frequent occurrence,
# if one exists.
def resample(data, timestep):

    # return the first non-zero mode unless the only mode is zero
    def mode(series):
        counts = series.value_counts()
        modes = counts.index
        if len(modes) == 1:
            return modes[0]
        if modes[0] == 0:
            return modes[1]
        return modes[0]
    # return = data.resample(args.timestep, how='median')
    return data.resample(timestep, how=mode)


# Concatenate the sensor value and activity label
# dataframes so that the new index is the union
# of the two and missing values are zero filled.
def combine(sensorValues, activityLabels):
    # df = pd.concat([sensorValues, activityLabels], axis=1)
    df = pd.concat([sensorValues, activityLabels], axis=1, join='inner')
    df.fillna(value=0, inplace=True)
    return df


# Extract a pandas time series from a file such as
# 'data/kasteren/2010/datasets/houseA/activity_labels.txt'
# Overwrite activity labels instead of having multiple
# activity labels for any timestep.
# Overwrites will be printed to stdout.
def activityLabels(fileName):
    with open(fileName, 'rb') as f:
        raw = pd.read_table(
            f,
            sep='\t',  # column separator
            engine='c',  # default value
            skiprows=[0, 1],  # disregard first two rows
            header=None,  # don't parse column names from header
            parse_dates=[0, 1],  # parse dates in first two cols
            names=['start', 'end', 'ID'],  # col names
        )
    s = pd.Series(
        np.nan,  # fill with NaN
        index=pd.date_range(
            start=raw['start'].min(),
            end=raw['end'].max(),
            freq='S',
        ),
    )
    overWrites = 0
    for index, row in raw.iterrows():
        # check for multiple activity labels for single time step
        try:
            assert s.loc[row['start']:row['end']].isnull().values.all()
        except:
            print('Overwriting unique values {} with {}'.format(
                s.loc[row['start']:row['end']].unique(),
                row['ID']))
            overWrites += 1
        s.loc[row['start']:row['end']] = row['ID']
    print('overwrites = {}'.format(overWrites))
    s = pd.DataFrame(
        s,
        columns=['activity'],
    )
    return s


# Given a file such as
# 'data/kasteren/2010/datasets/houseA/sensor_values.txt'
# and a list of possible sensor IDs
# Extract a pandas dataframe of size N by M
# where N is the number of seconds between and including
# the earliest and latest events
# and M is the number of sensors.
def sensorValues(fileName, sensorIDs):
    with open(fileName, 'rb') as f:
        raw = pd.read_table(
            f,
            sep='\t',  # column separator
            engine='c',  # default value
            skiprows=[0, 1],  # disregard first two rows
            header=None,  # don't parse column names from header
            parse_dates=[0, 1],  # parse dates in first two cols
            names=['start', 'end', 'ID', 'value'],  # col names
        )
    df = pd.DataFrame(
        np.nan,  # fill with NaN
        index=pd.date_range(
            start=raw['start'].min(),
            end=raw['end'].max(),
            freq='S',
        ),
        columns=sensorIDs,
    )
    for index, row in raw.iterrows():
        df.loc[row['start']:row['end'], row['ID']] = row['value']
    return df


# Given a file such as
# 'data/kasteren/2010/datasets/houseA/sensor_names.txt'
# return a map of sensor IDS to names.
def sensorNames(fileName):
    names = {}
    lineNum = 0
    pattern = r'([0-9]{1,2})\s+\'([^\']+)\''
    with open(fileName, 'rb') as f:
        for line in f:
            lineNum += 1
            match = re.search(pattern, line)
            names[int(match.group(1))] = match.group(2)
    return names


# Given a file such as
# 'data/kasteren/2010/datasets/houseA/activity_names.txt'
# return a map of activity IDS to names.
def activityNames(fileName):
    names = {}
    names[0] = 'idle'
    lineNum = 0
    with open(fileName, 'rb') as f:
        for line in f:
            lineNum += 1
            line = line.strip()[1:-1]
            if len(line) == 0:
                continue
            names[lineNum] = line
    return names


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Parse Kasteren data into canonical form. '
                     'Written in Python 2.7.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-A', '--activity_names',
                        default=('../../data/kasteren/2010/datasets/houseA/'
                                 'activity_names.txt'),
                        help='File containing acitivity names.')
    parser.add_argument('-a', '--activity_labels',
                        default=('../../data/kasteren/2010/datasets/houseA/'
                                 'activity_labels.txt'),
                        help='File containing activity labels.')
    parser.add_argument('-S', '--sensor_names',
                        default=('../../data/kasteren/2010/datasets/houseA/'
                                 'sensor_names.txt'),
                        help='File containing sensor names.')
    parser.add_argument('-s', '--sensor_values',
                        default=('../../data/kasteren/2010/datasets/houseA/'
                                 'sensor_values.txt'),
                        help='File containing sensor values.')
    parser.add_argument('-D', '--data',
                        default='./data.csv.gz',
                        help=('Time series of sensor values and activity '
                              'labels.'))
    parser.add_argument('-r', '--resampled_data',
                        default='./resampled_data.csv.gz',
                        help=('Resampled time series of sensor values and '
                              'activity labels.'
                              'Resampling results in the non-zero majority '
                              'unless zero is the only value.'))
    parser.add_argument('-n', '--names',
                        default='./names.json',
                        help='Activity and sensor name maps in JSON.')
    parser.add_argument('-t', '--timestep',
                        default='T',
                        help=('Used to downsample. '
                              'Time step given by a pandas time series '
                              'frequency: "T"=minute, "S"=second.'))
    return parser.parse_args()

if __name__ == '__main__':
    main()
