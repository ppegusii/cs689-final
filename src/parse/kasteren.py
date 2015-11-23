#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
import re
import sys


def main():
    args = parseArgs(sys.argv)
    print(args)
    aNames = activityNames(args.activity_names)
    print(aNames)
    sNames = sensorNames(args.sensor_names)
    print(sNames)
    sValues = sensorValues(args.sensor_values, sNames.keys())
    print(sValues)
    aLabels = activityLabels(args.activity_labels)
    print(aLabels)


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
    return s


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


def activityNames(fileName):
    names = {}
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
        description='Gate classifier. Written in Python 2.7.',
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
    parser.add_argument('-x',
                        default='./x',
                        help='Not yet used. Time series of sensor values.')
    parser.add_argument('-y',
                        default='./y',
                        help='Not yet used. Time series of activity labels.')
    parser.add_argument('-d', '--delta',
                        default='T',
                        help=('Not yet used. '
                              'Will be used to downsample. '
                              'Time step given by a pandas time series '
                              'frequency: "T"=minute, "S"=second.'))
    return parser.parse_args()

if __name__ == '__main__':
    main()
