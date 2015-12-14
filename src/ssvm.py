import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM, NSlackSSVM,OneSlackSSVM
import argparse
import load
import split
import sys

#Using pystruct
def test_SSVM(X_train, X_test, y_train, y_test):
    #Train using linear chain CRF
    #https://groups.google.com/forum/#!topic/pystruct/KIkF7fzCyDI

    model = ChainCRF()
    #ssvm = NSlackSSVM(model=model, C=.1, max_iter=11) # almost similar to FrankWolfeSSVM
    ssvm = FrankWolfeSSVM(model=model, C=0.01, max_iter=11)
    # c=0.2 -> 62.86 % accuracy <==> c=0.1

    #ssvm = OneSlackSSVM(model=model) #doesn't work as well
    ssvm.fit(X_train, y_train)
    print "Learning complete..."
    print("Test score with chain CRF: %f" % ssvm.score(X_test, y_test))

def replace(l, label_map):
    return map(lambda x: label_map[int(x)], l)

def main():
    args = parseArgs(sys.argv)
    data = load.data(args.data)
    classify(data)

def relabel(data):
    y = data['activity']

    labels = np.array(np.unique(y), dtype=int)
    labels_map = dict( zip(labels, range(len(labels))))
    y = replace(y, labels_map)
    data['activity'] = y
    return data


def classify(data):
    data = relabel(data)

    trainDf, testDf, trainLens, testLens, testFrac = split.trainTest(
        data, 5400, 5400*2, testSize=0.3)

    # e.g. structure of the array
    # X = [np.array([  [f1],[f2],[f3 ] ... [ N days], dtype=uint8 )]
    # Y = [np.array([   a, b , c])]
    # splitting so that we get a fraction of the day for training the labels
    X_train = np.array_split(trainDf.values[:, :trainDf.shape[1] - 2], 100)
    y_train = np.array_split(trainDf.values[:, trainDf.shape[1] - 1], 100)

    X_test = np.array_split(np.array(testDf.values[:, :testDf.shape[1] - 2], dtype=np.uint8), 30)
    y_test = np.array_split(np.array(testDf.values[:, testDf.shape[1] - 1], dtype=np.uint8), 30)


    test_SSVM(X_train, X_test, y_train, y_test)
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
