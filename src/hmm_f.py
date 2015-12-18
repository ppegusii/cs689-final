#!/usr/bin/env python

# using the CRF suite to create the prediction model

import argparse
import load
import split
import sys
import numpy as np
from seqlearn.hmm import MultinomialHMM

from sklearn.cross_validation import KFold, StratifiedKFold
import json
import gzip


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

    for house in [  'A', 'B' , 'C']:
        for f in ['data', 'last', 'change']:
            loc = data_loc_str.format(house=house,feature=f)
            #load the data
            data = load.data(loc, dtype_str=False)
            res_obj = {"y_pred":[], "y_true":[], "acc":[]}

            #split data into training and testing
            #trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
            #    data, 5400, 5400*2, testSize=0.3)



            #### WITHOUT CROSSVALID
            #X_test = np.array_split(np.array(testDf.values[:, :testDf.shape[1] - 2], dtype=np.uint8),2)
            #y_test = np.array_split(np.array(testDf.values[:, testDf.shape[1] - 1], dtype=np.uint8),2)

            #test_CRF(X_train, X_test, y_train, y_test, house, f,999)
            #exit()
            #### WITHOUT CROSSVALID
            #kfold
            X_train = np.array(np.array_split(data.values[:, :data.shape[1] - 2], 10))
            y_train = np.array(np.array_split(data.values[:, data.shape[1] - 1], 10))
            kf = KFold(len(X_train), n_folds=5)
            #kfold

            #strat
            #X_train = np.array(data.values[:, :data.shape[1] - 2])
            #y_train = np.array(data.values[:, data.shape[1] - 1])
            #kf = StratifiedKFold(data['activity'], n_folds=5)
            #strat

            accuracies = []
            # cross validation
            for i, (train_index, test_index) in enumerate(kf):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train1, X_test1 = X_train[train_index], X_train[test_index]
                y_train1, y_test1 = y_train[train_index], y_train[test_index]


                #stratfied
                #X_train1 = np.array_split(X_train1, 100)
                #X_test1 = np.array_split(X_test1, 10)
                #y_train1 = np.array_split(y_train1, 100)
                #y_test1 = np.array_split(y_test1, 10)
                #strat

                clf = train_HMM(X_train1, y_train1, house, f, i)

                accuracy, y_pred, y_true = test_HMM(clf, X_test1, y_test1, house, f, i)

                obj = {"y_pred":y_pred.tolist(), "y_true":y_true.tolist(), "acc":accuracy}
                #write the results:
                with gzip.open('hmm_model_f/hmm_' + house + f + str(i)+ '.json.gz', 'w') as out:
                    json.dump(obj, out)

                #clfs.append(clf)
                accuracies.append(accuracy)
                res_obj['y_pred'].append(y_pred.tolist())
                res_obj['y_true'].append(y_true.tolist())
                res_obj['acc'].append(accuracy)

            print accuracies
            with gzip.open('hmm_model_f/hmm_' + house + f +'_all.json.gz', 'w') as out:
                json.dump(res_obj, out)

def train_HMM(X_train, y_train, house, f,i ):
    #X_train = [s2features(s) for s in X_train]
    clf = MultinomialHMM(decode='viterbi')
    trainLens = np.array(map(lambda x: len(x), X_train))

    X_train = np.array(np.concatenate(X_train))
    y_train = np.array(np.concatenate(y_train))


    print(X_train)
    print(len(X_train))
    print(X_train[0].shape)
    print(len(y_train))
    print(y_train[0])
    print(trainLens, sum(trainLens))

    clf.fit(X_train, y_train, trainLens)

    #model_name = 'crf_models/house_' + house + '_'+ f + str(i) + '.crfsuite'
    #trainer.train(model_name)
    print str(i),'. House:', house, '. Feature: ', f, ' training complete.'
    return clf

def test_HMM(clf, X_test, y_test, house, f, i ):
    #model_name = 'crf_models/house_' + house + '_'+ f + str(i) + '.crfsuite'

    #X_test = [s2features(s) for s in X_test]
    testLens = np.array(map(lambda x: len(x), X_test))
    print testLens
    X_test = np.array(np.concatenate(X_test))
    y_test = np.array(np.concatenate(y_test))

    y_pred = clf.predict(X_test, testLens)
    y_test = np.array(y_test, dtype=np.uint8)
    accuracy = sum(y_pred == y_test)*1./len(y_pred)

    print('Accuracy: ', accuracy)
    #print y_pred
    return accuracy, y_pred, y_test

if __name__ == '__main__':
    main()
