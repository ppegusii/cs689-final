#!/usr/bin/env python
import argparse
import load
import split
import sys


def main():
    data = load.data(args.data)

    train_df, test_df, train_lens, test_lens, test_frac = split.trainTest(data, 5400,5400, testSize=0.3)

    X_train = train_df.values[:, :train_df.shape[1] - 2]
    y_train = train_df.values[:, train_df.shape[1] - 1]
    X_test = test_df.values[:, :test_df.shape[1] - 2]
    y_test = test_df.values[:, test_df.shape[1] - 1]


