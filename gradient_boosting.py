import sys

from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
import time

from classifier import MyClassifier
from util import load_pickled_dataset


__author__ = 'junjiah'


class MyGradientBoosting(MyClassifier):
    def __init__(self):
        self.gradient_boosting = None

    def train(self, data_path='data/train.pkl', n_estimators=10, learning_rate=0.1):
        labels, instances = load_pickled_dataset(data_path)
        start_time = time.clock()
        self.gradient_boosting = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate,
                                                            n_estimators=n_estimators, subsample=0.3,
                                                            min_samples_split=2,
                                                            min_samples_leaf=1,
                                                            max_depth=3,
                                                            init=None,
                                                            random_state=None,
                                                            max_features=None,
                                                            verbose=2)
        self.gradient_boosting.fit(instances, labels)
        end_time = time.clock()
        print "STATUS: model training done. elapsed time - %d seconds" % (end_time - start_time)
        print "INFO: " + str(self.gradient_boosting)

    def predict(self, data_path='data/test.pkl'):
        labels, instances = load_pickled_dataset(data_path)
        return self.gradient_boosting.predict(instances)

    def save(self, file_path='model/gbc_model'):
        joblib.dump(self.gradient_boosting, file_path)

    def load(self, file_path='model/gbc_model'):
        self.gradient_boosting = joblib.load(file_path)

    def write_results(self, predictions):
        super(MyGradientBoosting, self).write(predictions, 'gbc_prediction.csv')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        GBC = MyGradientBoosting()
        GBC.train()
    else:
        print 'INFO: so you just wanna load some functions? ok...'