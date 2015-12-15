import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM,OneSlackSSVM
import argparse
import load
import split
import sys
from sklearn.cross_validation import KFold
import pickle

def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data)

     #split data into training and testing
    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)

    classify(data)


# Using pystruct
def test_SSVM(X_train, X_test, y_train, y_test):

    #print X_train.shape, X_train[0].shape

    # splitting the 8 sub-arrays into further:
    #X_train = np.concatenate([np.array_split(x, 100) for x in X_train])
    #y_train = np.concatenate([np.array_split(y, 100) for y in y_train])

    #X_test = np.concatenate([np.array_split(x, 30) for x in X_test])
    #y_test = np.concatenate([np.array_split(y, 30) for y in y_test])

    #print X_train.shape
    #print X_train[0].shape
    #print y_train[0].shape
    #exit()
    #Train using linear chain CRF
    #https://groups.google.com/forum/#!topic/pystruct/KIkF7fzCyDI

    model = ChainCRF()
    #ssvm = NSlackSSVM(model=model, C=.1, max_iter=11) # almost similar to FrankWolfeSSVM
    ssvm = FrankWolfeSSVM(model=model, C=0.1, max_iter=11)
    # c=0.2 -> 62.86 % accuracy <==> c=0.1

    #ssvm = OneSlackSSVM(model=model) #doesn't work as well
    ssvm.fit(X_train, y_train)
    print "Learning complete..."
    accuracy = ssvm.score(X_test, y_test)
    print("Test score with chain CRF: %f" % accuracy )

    return ssvm, accuracy

# used to replace the labels with the new labels that starts with 0;
def replace(l, label_map):
    return map(lambda x: label_map[int(x)], l)

def relabel(data):
    y = data['activity']

    labels = np.array(np.unique(y), dtype=int)
    labels_map = dict( zip(labels, range(len(labels))))
    y = replace(y, labels_map)
    data['activity'] = y
    print np.unique(y)
    return data


def classify(data):
    data = relabel(data)

    # dividing the data into training and testing
    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)

    # e.g. structure of the array
    # X = [np.array([  [f1],[f2],[f3 ] ... [ N days], dtype=uint8 )]
    # Y = [np.array([   a, b , c])]
    # splitting so that we get a fraction of the day for training the labels
    X_train = np.array(np.array_split(trainDf.values[:, :trainDf.shape[1] - 2], 10))
    y_train = np.array(np.array_split(trainDf.values[:, trainDf.shape[1] - 1], 10))

    # test dataset - dividing into subsequences
    X_test = np.array(np.array_split(testDf.values[:, :testDf.shape[1] - 2], 30))
    y_test = np.array(np.array_split(testDf.values[:, testDf.shape[1] - 1], 30))
    print np.unique(np.concatenate(y_train).ravel())
    print np.unique(np.concatenate(y_test).ravel())

    #test_SSVM(X_train, X_test, y_train, y_test)
    #exit()
    # 5 fold validation;
    kf = KFold(len(X_train), n_folds=5)

    clfs = []
    accuracies = []
    # cross validation
    for train_index, test_index in kf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]
        print X_train1.shape,  y_train1.shape
        print X_train1[0].shape,  y_train1[0].shape

        clf, accuracy = test_SSVM(X_train1, X_test1, y_train1, y_test1)
        clfs.append(clf)
        accuracies.append(accuracy)

    print accuracies

    ssvm = clfs[np.argmax(accuracies)]
    print "Learning complete..."
    accuracy = ssvm.score(X_test, y_test)
    print("Test score with chain CRF: %f" % accuracy )


    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_true, y_pred)

    output = open('ssvm_last.pkl', 'wb')
    pickle.dump(clfs, output)


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
