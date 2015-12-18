import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM,OneSlackSSVM
import argparse
import load
import split
import sys
from sklearn.cross_validation import KFold, StratifiedKFold
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import json
import gzip

data_loc_str = '../data/kasteren/2010/datasets/house{house}/{feature}.csv.gz'

def main():
    #sample: {"y_pred":[], "y_true":[], "acc":[]}
    #args = parseArgs(sys.argv)
    for house in [ 'A', 'B' , 'C']:
        for f in ['lastchange']:#,'last', 'change', 'data']:
            loc = data_loc_str.format(house=house,feature=f)
            data = load.data(loc)
            classify(data, house, f)



# Using pystruct
def train_SVM(X_train, y_train):
    svm = LinearSVC(dual=False, C=.1)
    svm.fit(np.vstack(X_train), np.hstack(y_train))

    #print("Test score with linear SVM: %f" % svm.score(np.vstack(X_test),
    #                                               np.hstack(y_test)))
    return svm


def test_SVM(svm, X_test, y_test):
    y_pred = svm.predict(np.vstack(X_test))
    #print len(y_pred), len(y_test)
    y_pred = np.hstack(y_pred)
    y_test = np.hstack(y_test)
    #print len(y_pred), len(y_test)
    accuracy = sum(y_pred == y_test)*1./len(y_pred)
    #print confusion_matrix(y_test, y_pred)

    #accuracy = ssvm.score(X_test, y_test)
    print("Test score with SVM: %f" % accuracy )

    return accuracy, y_pred, y_test


def classify(data, house, f):
    res_obj = {"y_pred":[], "y_true":[], "acc":[]}

    X_train = np.array(np.array_split(data.values[:, :data.shape[1] - 2], 10))
    y_train = np.array(np.array_split(data.values[:, data.shape[1] - 1], 10))

    kf = KFold(len(X_train), n_folds=5)

    clfs = []
    accuracies = []
    # cross validation
    for i, (train_index, test_index) in enumerate(kf):

        print("TRAIN:", train_index, "TEST:", test_index)
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]


        #print X_train1.shape,  y_train1.shape
        #print X_train1[0].shape,  y_train1[0].shape
        clf = train_SVM(X_train1, y_train1)
        accuracy, y_pred, y_true = test_SVM(clf, X_test1, y_test1)
        #save the model
        output = open('svm_models/svm_' + house + f + str(i)+ '.pkl', 'wb')
        pickle.dump(clf, output)


        obj = {"y_pred":y_pred.tolist(), "y_true":y_true.tolist(), "acc":accuracy}
        #write the results:
        with gzip.open('svm_models/svm_' + house + f + str(i)+ '.json.gz', 'w') as out:
            json.dump(obj, out)

        clfs.append(clf)

        res_obj['y_pred'].append(y_pred.tolist())
        res_obj['y_true'].append(y_true.tolist())
        res_obj['acc'].append(accuracy)
        #accuracies.append(accuracy)

    print 'House:', house, 'Feature:', f,
    print res_obj['acc']
    with gzip.open('svm_models/svm_' + house + f +'_all.json.gz', 'w') as out:
        json.dump(res_obj, out)

    #ssvm = clfs[np.argmax(accuracies)]
    #print "Learning complete..."
    #accuracy = ssvm.score(X_test, y_test)
    #print("Test score with chain CRF: %f" % accuracy )


    print "Learning SVM complete."

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


if __name__=="__main__":
    main()
