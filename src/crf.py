import numpy as np
import pandas as pd
from parse import load_sensordata
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM,NSlackSSVM,OneSlackSSVM


def test_SVM(X_train, X_test, y_train, y_test):
    svm = LinearSVC(dual=False, C=.1)
    svm.fit(np.vstack(X_train), np.hstack(y_train))
    print("Test score with linear SVM: %f" % svm.score(np.vstack(X_test),
                                                   np.hstack(y_test)))

def replace(l, label_map):
    return map(lambda x: [label_map[int(x[0])]], l)

def main():

    house = 'A'

    df = load_sensordata(house)
    # convert to features and labels
    X = df.as_matrix(range(3,16))#[:10000]
    y = df.as_matrix([16])#[:10000]
    print y
    labels = np.array(np.unique(y), dtype=int)
    labels_map = dict( zip(labels, range(len(labels))))
    y = replace(y, labels_map)
    #print y
    #print np.unique(y)


    # y = y
    # print max(y)
    # print min(y)
    X, y = np.array(X, dtype=np.uint8),np.int_(y) # np.array(y, dtype=np.uint8)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.7, random_state=0)

    # Training using Linear SVM
    test_SVM(X_train, X_test, y_train, y_test)
    exit()
    # Y should be integer and starting from 0. https://github.com/pystruct/pystruct/issues/114
    X_train = [  np.array(x_i, dtype=np.uint8) [np.newaxis,:] for x_i in X_train ]
    X_test =  [  np.array(x_i, dtype=np.uint8)[np.newaxis, :] for x_i in X_test ]
    y_train = np.array([  np.array(y_i) for y_i in y_train ])
    y_test = np.array([  np.array(y_i) for y_i in y_test ])

    #idx  = np.argmax(y_train)

    print np.unique(y_train.ravel())
    #print y_train[idx-5: idx+5]
    #print max(y_train)

    #print X_train[0]
    #print X_train[0].shape
    #print y_train[0]
    #print y_train[0].shape
    # print y_train[0].dtype, X_train[0].dtype

    print "Learning SVM complete."
    #Train using linear chain CRF
    #https://groups.google.com/forum/#!topic/pystruct/KIkF7fzCyDI
    model = ChainCRF()
    #ssvm = NSlackSSVM(model=model, C=1, max_iter=11)
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=11)

    #ssvm = OneSlackSSVM(model=model)
    ssvm.fit(X_train, y_train)
    print "Learning complete..."
    print("Test score with chain CRF: %f" % ssvm.score(X_test, y_test))




if __name__=="__main__":
    main()
