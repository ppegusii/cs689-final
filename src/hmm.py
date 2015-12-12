#!/usr/bin/env python
from __future__ import print_function
import argparse
import load
from seqlearn.hmm import MultinomialHMM
import split
import sys


def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data)
    classify(data)


def classify(data):
    # trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #     data, 86400, 86400*2, testSize=0.3)
    # trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #     data, 21600, 21600*2, testSize=0.3)
    # trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #     data, 10800, 10800*2, testSize=0.3)
    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)
    # trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #     data, 5400, 5400*2, testSize=0.1)
    # trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #     data, 3600, 3600*2, testSize=0.3)
    print('Training sequences: {}'.format(len(trainLens)))
    print('Test sequences: {}'.format(len(testLens)))
    print('Portion of observations for testing: {}'.format(testFrac))
    X_train = trainDf.values[:, :trainDf.shape[1] - 2]
    y_train = trainDf.values[:, trainDf.shape[1] - 1]
    X_test = testDf.values[:, :testDf.shape[1] - 2]
    y_test = testDf.values[:, testDf.shape[1] - 1]
    clf = MultinomialHMM(decode='viterbi', alpha=0.01)
    clf.fit(X_train, y_train, trainLens)
    y_pred = clf.predict(X_test, testLens)
    compare = zip(y_test, y_pred)
    correct = [x for x in compare if x[0] == x[1]]
    print('Accuracy: {}'.format(float(len(correct))/len(compare)))


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='HMM. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        default=('../data/kasteren/2010/datasets/houseA/'
                                 'data.csv.gz'),
                        help=('Time series of sensor values and activity '
                              'labels.'))
    return parser.parse_args()


if __name__ == '__main__':
    main()
