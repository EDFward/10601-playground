import cPickle
import sys

import numpy as np
from scipy.io.arff import loadarff


__author__ = 'junjiah'

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
LABEL_MAP = {v: k for k, v in dict(enumerate(CLASSES, 1)).items()}


def load_arff_dataset(file_path):
    """
    Wrapper method for arff loading
    :param file_path: arff file path
    :return: labels and corresponding instances
    """
    data, meta = loadarff(open(file_path, 'r'))
    labels = np.asarray([LABEL_MAP[name] for name in data['class']], dtype=np.int32)
    instances = np.array(data[meta.names()[:-1]])
    instances = np.asarray(instances.tolist(), dtype=np.float32)
    print "STATUS: training data loading done. size %d * %d" % (len(instances), len(instances[0]))
    return labels, instances


def load_pickled_dataset(file_path, whitening=True):
    """
    Wrapper method for pickled data loading
    :param file_path: pickled file path
    :return: labels and corresponding instances
    """
    with open(file_path, 'rb') as f:
        instances = cPickle.load(f)
        labels = cPickle.load(f)
    if not whitening:
        return labels, instances
    else:
        return labels, whitening_ZCA(instances)


def whitening_ZCA(instances):
    epsilon = 10e-5
    normalized_instances = instances - instances.mean(axis=0)
    U, S, V = np.linalg.svd(np.dot(normalized_instances.T, normalized_instances) /
                            normalized_instances.shape[0])
    whitened_pcs = U / np.sqrt(S + epsilon)
    whitened_Z = np.dot(np.dot(normalized_instances, whitened_pcs), U.T)
    return whitened_Z


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: %s data_file' % sys.argv[0])

    labels, instances = load_pickled_dataset(sys.argv[1])
