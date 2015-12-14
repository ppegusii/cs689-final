#!/usr/bin/env python
from __future__ import print_function
import argparse
import gzip
import numpy as np
import pandas as pd
import sys

import load


def main():
    args = parseArgs(sys.argv)
    raw = load.data(args.input)
    rep = None
    if args.representation == 'last':
        rep = last(raw)
    elif args.representation == 'change':
        rep = change(raw)
    else:
        print('Invalid representation requested: {}'.format(
            args.representation))
        sys.exit(1)
    write(rep, args.output)


def last(df):
    # XOR with self shifted by 1 row (change)
    # Find the indices that have value 1
    # Forward fill the rows between the indices

    feat = df.iloc[:, :-1].values
    # print('feat.shape = {}'.format(feat.shape))
    # create the shifted sequence by copying the first row and deleting the last
    # print('feat[0:1, :].shape = {}'.format(feat[0:1, :].shape))
    # print('feat[:-1, :].shape = {}'.format(feat[:-1, :].shape))
    shifted = np.concatenate([feat[0:1, :], feat[:-1, :]])
    change = np.logical_xor(feat, shifted).astype(int)
    idxs = np.where(change)
    idxs, inv = np.unique(idxs[0], return_inverse=True)
    # print('idxs = {}'.format(idxs))
    if len(idxs) != len(inv):
        print('{} simultaneous sensor firings occurred!'.format(
              len(inv) - len(idxs)))
    for i in xrange(len(idxs)):
        # print('*****')
        # print(change)
        if i != len(idxs) - 1:
            change[idxs[i]+1:idxs[i+1], :] = change[idxs[i]:idxs[i]+1, :]
        else:
            change[idxs[i]+1:, :] = change[idxs[i]:idxs[i]+1, :]
        # print(change)
    newDf = df.copy()
    newDf.iloc[:, :-1] = change
    return newDf


def change(df):
    # XOR with self shifted by 1 row (change)

    feat = df.iloc[:, :-1].values
    # create the shifted sequence by copying the first row and deleting the last
    shifted = np.concatenate([feat[0:1, :], feat[:-1, :]])
    change = np.logical_xor(feat, shifted).astype(int)
    newDf = df.copy()
    newDf.iloc[:, :-1] = change
    return newDf


def testChange():
    # From paper
    data = pd.DataFrame(
        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0],
                  [0, 1, 0], [0, 0, 0]]))
    result = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0],
                       [1, 0, 0], [0, 1, 0]])
    print('Input')
    print(data.values)
    output = change(data).values
    print('Output')
    print(output)
    print('True')
    print(result)
    assert np.array_equal(output, result)


def testLast():
    # From paper
    data = pd.DataFrame(
        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0],
                  [0, 1, 0], [0, 0, 0]]))
    result = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                       [1, 0, 0], [0, 1, 0]])
    print('Input')
    print(data.values)
    output = last(data).values
    print('Output')
    print(output)
    print('True')
    print(result)
    assert np.array_equal(output, result)


def write(df, fileName):
    with gzip.open(fileName, 'w') as f:
        df.to_csv(f, float_format='%.0f')


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Parse Kasteren data into canonical form. '
                     'Written in Python 2.7.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input',
                        help='CSV of sequence in Raw format.')
    parser.add_argument('output',
                        help='Path of CSV output.')
    parser.add_argument('-r', '--representation',
                        default='last',
                        help=('Feature representation of output. '
                              'Choices {last, change}.'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
