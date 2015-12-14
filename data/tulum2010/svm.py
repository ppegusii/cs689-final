from sklearn import cross_validation
from sklearn import svm
from numpy import genfromtxt
my_data = genfromtxt('/Users/dongchen/Downloads/sample_tulum2010.dat', delimiter=',')


X = my_data[:,1:36]
y = my_data[:,37]

print(len(X),len(y))

X = X[0:86400*10]
y = y[0:86400*10]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.7, random_state=0)

#clf = svm.SVC()
clf = svm.LinearSVC()

clf.fit(X_train, y_train)
clf.predict(X_test)

print clf.score(X_test,y_test)
