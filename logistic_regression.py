import numpy
import sys
import theano
import theano.tensor as T
import time
from classifier import MyClassifier
from util import load_pickled_dataset

FEATURE_NUMBER = 3072
CLASS_NUMBER = 10


def load_theano_dataset(file_path):
    """
    Transform arff data to theano shared variables
    :param file_path: arff file path
    :return: x and y in shared variable form
    """
    data_y, data_x = load_pickled_dataset(file_path)
    # change y to be 0~9
    for i in range(len(data_y)):
        data_y[i] -= 1

    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')


class MyLogisticRegression(MyClassifier):
    """ Multi-class logistic regression class """

    def __init__(self, input=None, n_in=FEATURE_NUMBER, n_out=CLASS_NUMBER):
        """
        Initialize parameters

        :param input: symbolic variable for the input
        :param n_in: number of input data dimensions (features)
        :param n_out: number of output data dimensions (classes)
        """
        if input is None:
            self.x = T.dmatrix('x')
        else:
            self.x = input

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out, ),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """
        Return mean of negative log-likelihood of the prediction
        of the model.
        :param y: a vector that gives each example the correct
        label
        :return: negative log likelihood
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return ratio of errors according to ground truth y
        :param y: a vector that gives each example the correct
        label
        :return: ratio of errors in whole predictions
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def train(self, learning_rate=0.001, batch_size=100, n_epochs=3000):
        """
        Use Stochastic Gradient Descent to train LR model
        :param learning_rate: learning rate
        :param batch_size: size of minibatch when doing SGD
        :param n_epochs: number of epochs
        :return: trained LR model
        """
        train_set_x, train_set_y = load_theano_dataset('data/train.pkl')

        y = T.ivector('y')
        index = T.iscalar()

        # self.classifier = MyLogisticRegression(input=x, n_in=FEATURE_NUMBER, n_out=CLASS_NUMBER)

        cost = self.negative_log_likelihood(y)

        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)

        updates = [(self.W, self.W - learning_rate * g_W),
                   (self.b, self.b - learning_rate * g_b)]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        print "STATUS: begin training the logistic regression model"

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        epoch = 0
        start_time = time.clock()

        while epoch < n_epochs:
            epoch += 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)

        end_time = time.clock()

        print "STATUS: training done. elapsed time - %d seconds" % (end_time - start_time)

    def predict(self):
        """
        Use trained LR classifier for predictions
        :return: predictions
        """
        test_set_x, test_set_y = load_theano_dataset('data/test.pkl')

        start_time = time.clock()

        test_model = theano.function(
            inputs=[],
            outputs=self.y_pred,
            givens=[(self.x, test_set_x)]
        )

        pred = test_model()

        end_time = time.clock()
        print "STATUS: prediction done. elapsed time - %d seconds" % (end_time - start_time)
        # traverse the predictions and add 1 for labels
        for i in range(len(pred)):
            pred[i] += 1
        return pred

    def save(self, file_path='./model/lr_model'):
        from cPickle import dump, HIGHEST_PROTOCOL

        with file('./model/lr_model', 'wb') as f:
            dump(self, f, protocol=HIGHEST_PROTOCOL)

    def load(self, file_path='./model/lr_model'):
        from cPickle import load

        with file(file_path, 'rb') as f:
            model = load(f)
        # update computation graph components
        self.b, self.W = model.b, model.W
        self.x, self.y_pred, self.p_y_given_x = model.x, model.y_pred, model.p_y_given_x

    def write_results(self, predictions):
        super(MyLogisticRegression, self).write(predictions, 'lr_prediction.csv')


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        x = T.dmatrix('x')
        LR = MyLogisticRegression(x)
        LR.train()
    else:
        print 'INFO: so you just wanna load some functions? ok...'