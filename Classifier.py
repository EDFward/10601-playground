__author__ = 'junjiah'


class MyClassifier(object):
    """
    Abstract class for all kinds of classifiers ;)
    """

    def train(self, **kwargs):
        """
        Train the model
        """
        raise NotImplementedError()

    def predict(self):
        """
        Use the trained model to make predictions on test.arff
        """
        raise NotImplementedError()

    def save(self, file_path):
        """ Dump the model """
        raise NotImplementedError()

    def load(self, file_path):
        """ Load the model from files """
        raise NotImplementedError()

    def write(self, predictions, file_path):
        """
        Write prediction results to files
        :param predictions: predictions
        :param file_path: file path
        """
        with open(file_path, 'wb') as result_file:
            result_file.write("Id,Category\n")
            for i, prediction in enumerate(predictions, 1):
                result_file.write("%d,%d\n" % (i, prediction))  # add 1, from 1-10
