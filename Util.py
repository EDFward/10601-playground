import numpy as np
import sys
from scipy.io.arff import loadarff

__author__ = 'junjiah'

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
label_map = {v: k for k, v in dict(enumerate(classes, 1)).items()}


def load_arff_dataset(file_path):
    """
    Wrapper class for arff loading
    :param file_path: arff file path
    :return: labels and corresponding instances
    """
    data, meta = loadarff(open(file_path, 'r'))
    labels = np.asarray([label_map[name] for name in data['class']], dtype=np.int32)
    instances = np.array(data[meta.names()[:-1]])
    instances = np.asarray(instances.tolist(), dtype=np.float32)
    print "STATUS: training data loading done. size %d * %d" % (len(instances), len(instances[0]))
    return labels, instances


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: %s data_file' % sys.argv[0])

    labels, instances = load_arff_dataset(sys.argv[1])