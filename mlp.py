from classifier import MyClassifier


class MyMultiLayerPerceptron(MyClassifier):
    def load(self, file_path):
        return super(MyMultiLayerPerceptron, self).load(file_path)

    def train(self, data_path='data/train.pkl', **kwargs):
        return super(MyMultiLayerPerceptron, self).train(**kwargs)

    def write(self, predictions, file_path):
        super(MyMultiLayerPerceptron, self).write(predictions, file_path)

    def predict(self, data_path='data/test.pkl'):
        return super(MyMultiLayerPerceptron, self).predict()

    def save(self, file_path):
        return super(MyMultiLayerPerceptron, self).save(file_path)