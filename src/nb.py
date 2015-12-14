#!/usr/bin/env python
from __future__ import print_function
import argparse
import load
from sklearn import cross_validation, grid_search
from sklearn.naive_bayes import BernoulliNB
import sys


def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data)
    classify(data)


def classify(data):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data.values[:, 0:data.shape[1] - 2],  # X features
        data.values[:, data.shape[1] - 1],  # y labels
        test_size=0.3,
        random_state=0,
    )
    params = {
        'alpha': [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    }
    nb = BernoulliNB()
    clf = grid_search.GridSearchCV(nb, params)
    clf.fit(X_train, y_train)
    print('clf.best_estimator_ = {:s}'.format(clf.best_estimator_))
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
