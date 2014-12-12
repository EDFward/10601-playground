from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

import sys
from classifier import MyClassifier
from feats_repr import load_pickled_dataset


__author__ = 'junjiah'


class MyAdaBoost(MyClassifier):
    def __init__(self):
        self.adaboost = None

    def train(self, data_path='data/train.pkl', n_estimators=100, n_jobs=8):
        labels, instances = load_pickled_dataset(data_path)

        self.adaboost = AdaBoostClassifier(n_estimators=n_estimators)
        self.adaboost.fit(instances, labels)

        print "STATUS: model training done. "
        print "INFO: " + str(self.adaboost)

    def predict(self, data_path='data/test.pkl'):
        labels, instances = load_pickled_dataset(data_path)
        return self.adaboost.predict(instances)

    def save(self, file_path='model/ab_model'):
        joblib.dump(self.adaboost, file_path)

    def load(self, file_path='model/ab_model'):
        self.adaboost = joblib.load(file_path)

    def write_results(self, predictions):
        super(MyAdaBoost, self).write(predictions, 'ab_prediction.csv')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        AB = MyAdaBoost()
        AB.train()
    else:
        print 'INFO: so you just wanna load some functions? ok...'