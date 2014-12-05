import cPickle
import numpy as np
import sys
from scipy.io.arff import loadarff

__author__ = 'junjiah'

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
LABEL_MAP = {v: k for k, v in dict(enumerate(CLASSES, 1)).items()}


def load_arff_dataset(file_path):
    """
    Wrapper class for arff loading
    :param file_path: arff file path
    :return: labels and corresponding instances
    """
    data, meta = loadarff(open(file_path, 'r'))
    labels = np.asarray([LABEL_MAP[name] for name in data['class']], dtype=np.int32)
    instances = np.array(data[meta.names()[:-1]])
    instances = np.asarray(instances.tolist(), dtype=np.float32)
    print "STATUS: training data loading done. size %d * %d" % (len(instances), len(instances[0]))
    return labels, instances


def load_pickled_dataset(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
        labels = cPickle.load(f)
    return labels, data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: %s data_file' % sys.argv[0])

    labels, instances = load_pickled_dataset(sys.argv[1])