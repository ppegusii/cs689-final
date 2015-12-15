#!/usr/bin/env python

# using the CRF suite to create the prediction model

import argparse
import load
import split
import sys
import pycrfsuite
import numpy as np

from sklearn.cross_validation import KFold

data_loc_str = '../data/kasteren/2010/datasets/house{house}/{feature}.csv.gz'
from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def sensor2features(s,i):
    sensor = s[i]
    #features = ['s_'+str(i)+'=' + str(bool(int(sensor[i]))) for i in range(len(sensor))]
    features = ['i_'+str(i)+'=' + str(sensor[i]) for i in range(len(sensor))]
    #features = np.append(features, features_neg)
    return features

def s2features(s):
    return [sensor2features(s, i) for i in range(len(s))]

def main():
    # args = parseArgs(sys.argv)

    for house in [ 'A', 'B' , 'C']:
        for f in ['last', 'change', 'data']:
            loc = data_loc_str.format(house=house,feature=f)
            #load the data
            data = load.data(loc, dtype_str=True)

            #split data into training and testing
            #trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
            #    data, 5400, 5400*2, testSize=0.3)


            X_train = np.array(np.array_split(data.values[:, :data.shape[1] - 2], 10))
            y_train = np.array(np.array_split(data.values[:, data.shape[1] - 1], 10))

            #### WITHOUT CROSSVALID
            #X_test = np.array_split(np.array(testDf.values[:, :testDf.shape[1] - 2], dtype=np.uint8),2)
            #y_test = np.array_split(np.array(testDf.values[:, testDf.shape[1] - 1], dtype=np.uint8),2)

            #test_CRF(X_train, X_test, y_train, y_test, house, f,999)
            #exit()
            #### WITHOUT CROSSVALID
            kf = KFold(len(X_train), n_folds=5)

            clfs = []
            accuracies = []
            # cross validation
            for i, (train_index, test_index) in enumerate(kf):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train1, X_test1 = X_train[train_index], X_train[test_index]
                y_train1, y_test1 = y_train[train_index], y_train[test_index]

                train_CRF(X_train1, y_train1, house, f, i)

                clf, accuracy = test_CRF(X_test1, y_test1, house, f, i)
                clfs.append(clf)
                accuracies.append(accuracy)

            print accuracies
            exit()

            #train_model(trainDf)
            #test_model(testDf)

def train_CRF(X_train, y_train, house, f,i ):
    X_train = [s2features(s) for s in X_train]
    trainer = pycrfsuite.Trainer( verbose=False)

    X_train = np.concatenate([np.array_split(x, 20) for x in X_train])
    y_train = np.concatenate([np.array_split(y, 20) for y in y_train])
    print len(X_train)

    print X_train[0].shape, y_train[0].shape

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'max_iterations': 11,  # stop earlier
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-1,  # coefficient for L2 penalty
        # include transitions that are possible, but not observed
        'feature.possible_transitions': False
    })

    model_name = 'crf_models/house_' + house + '_'+ f + str(i) + '.crfsuite'
    trainer.train(model_name)
    print str(i),'. House:', house, '. Feature: ', f, ' training complete.'


def test_CRF( X_test, y_test, house, f, i ):
    model_name = 'crf_models/house_' + house + '_'+ f + str(i) + '.crfsuite'

    X_test = [s2features(s) for s in X_test]

    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)

    y_pred = np.array([])
    for xseq in X_test:
        y_pred = np.append(y_pred, np.array(tagger.tag(xseq), dtype=np.uint8))

    y_test = np.array(np.concatenate(y_test), dtype=np.uint8)


    accuracy = sum(y_pred == y_test)*1./len(y_pred)

    print('Accuracy: ', accuracy)
    #print y_pred
    return house+f+str(i), accuracy

def print_info(tagger):
    info = tagger.info()

    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])

    return house+f, accuracy


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
