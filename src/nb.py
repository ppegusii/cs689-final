#!/usr/bin/env python
from __future__ import print_function
import argparse
import load
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
import sys


def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data)
    classify(data)


def classify(data):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data.values[:, 0 : data.shape[1] - 2],  # X features
        data.values[:, data.shape[1] - 1],  # y labels
        test_size=0.7,
        random_state=0,
    )
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    scores = cross_validation.cross_val_score(
        clf,
        data.values[:, 0 : data.shape[1] - 2],
        data.values[:, data.shape[1] - 1],
        cv=10,
    )
    print(scores)
    print(scores.mean())


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='Naive Bayes. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        default=('../data/kasteren/2010/datasets/houseA/'
                                 'data.csv.gz'),
                        help=('Time series of sensor values and activity '
                              'labels.'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
