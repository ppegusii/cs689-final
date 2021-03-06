#!/usr/bin/env python
from __future__ import print_function
import argparse
import gzip
import itertools
import json
import load
import numpy as np
import pandas as pd
import pickle
from seqlearn.hmm import MultinomialHMM
import split
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
    # clf = MultinomialHMM(decode='viterbi', alpha=0.01)
    clf, accuracies = gridSearch(
        trainDf,
        trainLens,
        # decodes=['viterbi', 'bestfirst'],
        decodes=['viterbi'],
        # alphas=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        alphas=[0.000001, 1],
        init_eq_anys=[True, False],
    )
    saveClf(clf, clfFileName)
    print('best_estimator = {:s}'.format(clf))
    # accuracies = crossValidate(clf, trainDf, trainLens)
    print('Cross validation accuracies: {}'.format(accuracies))
    print('Mean cross validation accuracy: {}'.format(accuracies.mean()))
    print('Standard deviation in cross validation accuracy: {}'.format(
        accuracies.std()))
    clf.fit(X_train, y_train, trainLens)
    y_pred = clf.predict(X_test, testLens)
    print('Accuracy: {}'.format(accuracy(y_test, y_pred)))
    # print('Predicted label counts: {}'.format(
    #   pd.Series(y_pred).value_counts()))
    # print('True label counts: {}'.format(pd.Series(y_test).value_counts()))
    saveResults(y_pred, y_test, accuracies, resultsFile)


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


def accuracy(y_test, y_pred):
    compare = zip(y_test, y_pred)
    correct = [x for x in compare if x[0] == x[1]]
    return float(len(correct))/len(compare)


def gridSearch(seqs, lens, decodes=[None], alphas=[None], init_eq_anys=[None]):
    maxAcc = 0.0
    maxAccs = None
    bestClf = None
    for d, a, i in itertools.product(*[decodes, alphas, init_eq_anys]):
        clf = MultinomialHMM(decode=d, alpha=a, init_eq_any=i)
        accs = crossValidate(clf, seqs, lens)
        meanAcc = accs.mean()
        if meanAcc > maxAcc:
            maxAcc = meanAcc
            maxAccs = accs
            bestClf = clf
    '''
    for decode in decodes:
        for alpha in alphas:
            clf = MultinomialHMM(decode=decode, alpha=alpha)
            accs = crossValidate(clf, seqs, lens)
            meanAcc = accs.mean()
            if meanAcc > maxAcc:
                maxAcc = meanAcc
                maxAccs = accs
                bestClf = clf
    '''
    return bestClf, maxAccs


def crossValidate(clf, seqs, lens, folds=4):
    # Perform cross validation as follows:
    # Loop through the sequences folds times.
    # On each loop, if the index of sequence mod folds equals the fold number
    # assign that sequence to the testing set, otherwise assign to training set.
    # Train and test the classifier and save the accuracy.
    accuracies = list()
    for i in xrange(folds):
        trainSeqs = list()
        trainLens = list()
        testSeqs = list()
        testLens = list()
        startIdx = 0
        for j in xrange(len(lens)):
            leng = lens[j]
            endIdx = startIdx + leng
            seq = seqs.iloc[startIdx:endIdx, :]
            startIdx = endIdx
            if j % folds == i:
                testSeqs.append(seq)
                testLens.append(leng)
            else:
                trainSeqs.append(seq)
                trainLens.append(leng)
        trainDf = pd.concat(trainSeqs)
        testDf = pd.concat(testSeqs)
        X_train = trainDf.values[:, :trainDf.shape[1] - 2]
        y_train = trainDf.values[:, trainDf.shape[1] - 1]
        X_test = testDf.values[:, :testDf.shape[1] - 2]
        y_test = testDf.values[:, testDf.shape[1] - 1]
        clf.fit(X_train, y_train, trainLens)
        y_pred = clf.predict(X_test, testLens)
        accuracies.append(accuracy(y_test, y_pred))
    return np.array(accuracies)


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='HMM. Written in Python 2.7.',
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
