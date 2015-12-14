#!/usr/bin/env python

# using the CRF suite to create the prediction model

import argparse
import load
import split
import sys
import pycrfsuite
import numpy as np

def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data, dtype_str=True)
    classify(data)
    test_model(data)

def classify(data):

    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)

    #X_train = trainDf.values[:, :trainDf.shape[1] - 2]
    #y_train = trainDf.values[:, trainDf.shape[1] - 1]

    X_train = np.array_split(trainDf.values[:, :trainDf.shape[1] - 2], 66290)
    y_train = np.array_split(trainDf.values[:, trainDf.shape[1] - 1], 66290)

    #X_test = np.array_split(np.array(testDf.values[:, :testDf.shape[1] - 2], dtype=np.uint8), 30)
    #y_test = np.array_split(np.array(testDf.values[:, testDf.shape[1] - 1], dtype=np.uint8), 30)

    trainer = pycrfsuite.Trainer(verbose=False)
    #trainer.append(X_train, y_train)
    # X_train.shape = [  [sentence feature [ word feature] ] ]
    # y_train.shape = [   [sentence [ word label] ]   ]
    for xseq, yseq in zip(X_train, y_train):
        #print xseq.shape, yseq.shape
        #print xseq , yseq
        trainer.append(xseq, yseq)


    trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 11,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
    })

    trainer.train('house_a.crfsuite')
    print "Model created"

def test_model(data):

    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)

    #X_test = testDf.values[:, :testDf.shape[1] - 2]
    #y_test = testDf.values[:, testDf.shape[1] - 1]

    # using subsequences, splitting the entire month data into days
    X_test = np.array_split(np.array(testDf.values[:, :testDf.shape[1] - 2], dtype=np.uint8), 30)
    y_test = np.array_split(np.array(testDf.values[:, testDf.shape[1] - 1], dtype=np.uint8), 30)


    tagger = pycrfsuite.Tagger()
    tagger.open('house_a.crfsuite')
    #print "Predict" , tagger.tag([X_test[5]])
    #print "Correct", y_test[5]
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    compare = zip(y_test, y_pred)
    correct = [x for x in compare if x[0] == x[1]]
    print('Accuracy: {}'.format(float(len(correct))/len(compare)))
    # TODO confusion matrix ;
    # classificaiton report:


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='CRF. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        default=('../data/kasteren/2010/datasets/houseA/'
                                 'data.csv.gz'),
                        help=('Time series of sensor values and activity '
                              'labels.'))
    return parser.parse_args()



if __name__ == '__main__':
    main()
