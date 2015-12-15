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

data_loc_str = '../data/kasteren/2010/datasets/house{house}/{feature}.csv.gz'

def main():
    #args = parseArgs(sys.argv)
    for house in [ 'C', 'B' , 'A']:
        for f in ['last', 'change', 'data']:
            loc = data_loc_str.format(house=house,feature=f)
            data = load.data(loc)
            classify(data, house, f)


# Using pystruct
def train_SSVM(X_train, y_train):

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

    return ssvm

def test_SSVM(ssvm, X_test, y_test):
    y_pred = ssvm.predict(X_test)
    #print len(y_pred), len(y_test)
    y_pred = np.hstack(y_pred)
    y_test = np.hstack(y_test)
    #print len(y_pred), len(y_test)
    accuracy = sum(y_pred == y_test)*1./len(y_pred)
    #print confusion_matrix(y_test, y_pred)

    #accuracy = ssvm.score(X_test, y_test)
    print("Test score with chain CRF: %f" % accuracy )



# used to replace the labels with the new labels that starts with 0;
def replace(l, label_map):
    return map(lambda x: label_map[int(x)], l)

def relabel(data):
    y = data['activity']

    labels = np.array(np.unique(y), dtype=int)
    labels_map = dict( zip(labels, range(len(labels))))
    y = replace(y, labels_map)
    data['activity'] = y
    #print np.unique(y)
    return data


def classify(data, house, f):
    data = relabel(data)
    # WWW
    # dividing the data into training and testing
    #trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
    #    data, 5400, 5400*2, testSize=0.3)

    # e.g. structure of the array
    # X = [np.array([  [f1],[f2],[f3 ] ... [ N days], dtype=uint8 )]
    # Y = [np.array([   a, b , c])]
    # splitting so that we get a fraction of the day for training the labels
    #X_train = np.array(np.array_split(trainDf.values[:, :trainDf.shape[1] - 2], 10))
    #y_train = np.array(np.array_split(trainDf.values[:, trainDf.shape[1] - 1], 10))

    # test dataset - dividing into subsequences
    #X_test = np.array(np.array_split(testDf.values[:, :testDf.shape[1] - 2], 30))
    #y_test = np.array(np.array_split(testDf.values[:, testDf.shape[1] - 1], 30))
    # WWW

    X_train = np.array(data.values[:, :data.shape[1] - 2])
    y_train = np.array(data.values[:, data.shape[1] - 1])

    #print X_train.shape
    #test_SSVM(X_train, X_test, y_train, y_test)
    #exit()
    # 5 fold validation;
    #label = np.unique(data['activity'])
    kf = StratifiedKFold(data['activity'], n_folds=5)

    clfs = []
    accuracies = []
    # cross validation
    for i, (train_index, test_index) in enumerate(kf):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]

        #print np.unique(np.concatenate(y_train1).ravel())
        #print np.unique(np.concatenate(y_test1).ravel())
        X_train1 = np.array_split(X_train1, 100)
        X_test1 = np.array_split(X_test1, 10)
        y_train1 = np.array_split(y_train1, 100)
        y_test1 = np.array_split(y_test1, 10)


        #print X_train1.shape,  y_train1.shape
        #print X_train1[0].shape,  y_train1[0].shape
        clf = train_SSVM(X_train1, y_train1)
        accuracy = test_SSVM(clf, X_test1, y_test1)
        #save the model

        output = open('ssvm_models/ssvm_' + house + f + + str(i)+ '.pkl', 'wb')
        pickle.dump(clf, output)

        clfs.append(clf)
        accuracies.append(accuracy)

    print 'House:', house, 'Feature:', f,
    print accuracies

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
