from sklearn import svm
from sklearn.externals import joblib

import sys
from classifier import MyClassifier
from feats_repr import load_pickled_dataset


__author__ = 'junjiah'


class MySupportVectorMachine(MyClassifier):
    def __init__(self):
        self.svm = None

    def train(self, data_path='data/train.pkl', kernel='rbf'):
        labels, instances = load_pickled_dataset(data_path)

        self.svm = svm.SVC(kernel=kernel)
        self.svm.fit(instances, labels)

        print "STATUS: model training done. "
        print "INFO: " + str(self.svm)

    def predict(self, data_path='data/test.pkl'):
        labels, instances = load_pickled_dataset(data_path)
        return self.svm.predict(instances)

    def save(self, file_path='model/svm_model'):
        joblib.dump(self.svm, file_path)

    def load(self, file_path='model/svm_model'):
        self.svm = joblib.load(file_path)

    def write_results(self, predictions):
        super(MySupportVectorMachine, self).write(predictions, 'svm_prediction.csv')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        SVM = MySupportVectorMachine()
        SVM.train()
    else:
        print 'INFO: so you just wanna load some functions? ok...'