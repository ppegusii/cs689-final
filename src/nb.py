#!/usr/bin/env python
from __future__ import print_function
import argparse
import gzip
import json
import load
from sklearn import cross_validation, grid_search
from sklearn.naive_bayes import BernoulliNB
import pickle
import sys


def main():
    args = parseArgs(sys.argv)
    if args.source == 'k':
        data = load.data(args.data)
    elif args.source == 't':
        data = load.tulum(args.data, dtype_str=True)
    else:
        print('Invalid data source specified: {}'.format(args.source))
        sys.exit(1)
    classify(data, args.clfFile, args.resultsFile)


def classify(data, clfFileName, resultsFile):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data.values[:, 0:data.shape[1] - 2],  # X features
        data.values[:, data.shape[1] - 1],  # y labels
        test_size=0.3,
        random_state=0,
    )
    params = {
        # 'alpha': [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        'alpha': [0.00001, 1]
    }
    nb = BernoulliNB()
    clf = grid_search.GridSearchCV(nb, params)
    clf.fit(X_train, y_train)
    print('clf.best_estimator_ = {:s}'.format(clf.best_estimator_))
    saveClf(clf, clfFileName)
    scores = cross_validation.cross_val_score(
        clf,
        X_train,
        y_train,
        cv=4,
    )
    print('Cross validation scores: {}'.format(scores))
    print('Mean cross validation score: {}'.format(scores.mean()))
    print('Standard deviation in cross validation score: {}'.format(
        scores.std()))
    print('Test score: {}'.format(clf.score(X_test, y_test)))
    y_pred = clf.predict(X_test)
    saveResults(y_pred, y_test, scores, resultsFile)


def saveClf(clf, fileName):
    with open(fileName, 'w') as f:
        pickle.dump(clf, f)


def saveResults(y_pred, y_true, accuracies, fileName):
    results = {
        'y_pred': y_pred.tolist(),
        'y_true': y_true.tolist(),
        'acc': accuracies.tolist()
    }
    with gzip.open(fileName, 'w') as f:
        json.dump(results, f)


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='Naive Bayes. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        default=('../data/kasteren/2010/datasets/houseA/'
                                 'data.csv.gz'),
                        help=('Time series of sensor values and activity '
                              'labels.'))
    parser.add_argument('-s', '--source',
                        default='k',
                        help=('Source of data Kaseteren or Tulum. {k, t}'))
    parser.add_argument('-c', '--clfFile',
                        help=('File to save best trained classifier.')),
    parser.add_argument('-r', '--resultsFile',
                        help=('File to save y_true, y_pred.')),
    return parser.parse_args()


if __name__ == '__main__':
    main()
