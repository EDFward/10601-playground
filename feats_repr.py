import cPickle

import numpy as np
from scipy.io.arff import loadarff
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib


__author__ = 'junjiah'

PIC_LENGTH = 32
CHANNEL_LENGTH = 1024
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
LABEL_MAP = {v: k for k, v in dict(enumerate(CLASSES, 1)).items()}


def load_arff_dataset(file_path):
    """
    Wrapper method for arff loading
    :param file_path: arff file path
    :return: labels and corresponding instances
    """
    data, meta = loadarff(file(file_path, 'r'))
    labels = np.asarray([LABEL_MAP[name] for name in data['class']], dtype=np.int32)
    instances = np.array(data[meta.names()[:-1]])
    instances = np.asarray(instances.tolist(), dtype=np.float32)
    print "STATUS: training data loading done. size %d * %d" % (len(instances), len(instances[0]))
    return labels, instances


def load_pickled_dataset(file_path, whitening=False):
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
        return labels, whitening(instances)


# def sample_patches(instances, patch_size):


def extract_patches(instances, patch_size=8):
    """
    Build patches from original 3072 features based on given patch size.
    :param instances: M*3072 instance matrix
    :param patch_size: length of size of patch, so the actual size is patch_size^2
    :return: extracted patch_size*patch_size patches. One row per patch
    """

    def block_shaped_flat(arr, nrows, ncols):
        """
        Return an array of shape (n, nrows * ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n flattened sub-blocks with
        each sub-block preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows * ncols))

    patches = []
    for instance in instances:
        # split to 32*32 red, green, blue channel matrices
        pic_in_channels = instance.reshape(3, PIC_LENGTH, PIC_LENGTH)
        # process each 32*32 matrix as patch_size*patch_size blocks and flatten them
        flattened_patches = map(lambda m: block_shaped_flat(m, patch_size, patch_size), pic_in_channels)
        # zip each block's values from 3 channels as a patch_size*patch_size*3 feature vectors
        concatenated_features = [np.concatenate(i) for i in zip(*flattened_patches)]
        # add to total patch list
        patches += concatenated_features
    # reshape as a (patch_number * m) by (patch_size*patch_size*3) matrix
    return np.reshape(patches, newshape=(len(patches), -1))


def convert_pic(scaled_img_vec, pic_length=PIC_LENGTH):
    """
    Convert a *pic_length* dimension instance to an picture
    :param scaled_img_vec: given instance in 3072 dimensions, ranges from 0~1
    :param pic_length:
    :return:
    """
    img_vec = scaled_img_vec * 255
    i = img_vec.astype(np.uint8).reshape(3, pic_length, pic_length)
    return np.asarray([np.asarray(cell, dtype=np.uint8).T for cell in zip(*i)])


def convert_pic_in_patch(patch_batch, patch_size=8):
    """
    Convert a bunch of patches to an picture
    :param patch_batch: patches to be combined, each with dimension patch_size
    :param patch_size:
    :return:
    """

    def chunks(l, n):
        """ Yield successive n-sized chunks from l. """
        for i in xrange(0, len(l), n):
            yield l[i:i + n]

    patch_pics = map(lambda v: convert_pic(v, patch_size), patch_batch)
    pic = np.vstack([np.hstack(ch) for ch in chunks(patch_pics, PIC_LENGTH / patch_size)])
    return pic


def feat_transform(instances, k_means, patch_size=8, stride=2):
    """
    Transform feature vectors using 3 channels to k-means representation
    combining quadrants in original images
    :param instances: original feature vectors
    :param k_means: k_means clustering model
    :return: transformed feature vector
    """
    if k_means.mean_distances is None:
        print 'Mean distances in K means model not available, abort.'
        return

    k = len(k_means.mean_distances)
    feats = []

    for instance in instances:
        pic_in_channels = instance.reshape(3, PIC_LENGTH, PIC_LENGTH)
        iter_num = (PIC_LENGTH / 2 - patch_size) / stride + 1
        feat = []
        for i in range(2):
            for j in range(2):
                start_x, start_y = i * PIC_LENGTH / 2, j * PIC_LENGTH / 2
                quadrant_patches = []
                for i1 in range(iter_num):
                    for j2 in range(iter_num):
                        patch_x, patch_y = start_x + i1 * stride, start_y + j2 * stride
                        patch = pic_in_channels[:, patch_x: patch_x + patch_size,
                                patch_y: patch_y + patch_size].reshape(-1)
                        quadrant_patches.append(patch)
                quadrant_feature = np.maximum(np.zeros(k),
                                              k_means.transform(quadrant_patches) - k_means.mean_distances).sum(axis=0)
                feat.extend(quadrant_feature)
        feats.append(feat)
    return np.asarray(feats)


def whitening(inst):
    """ ZCA whitening. """
    epsilon, mu = 10e-5, inst.mean(axis=0)
    norm_inst = inst - mu
    cov = (np.dot(norm_inst.T, norm_inst) / norm_inst.shape[1])
    U, S, VT = np.linalg.svd(cov)

    tmp = np.dot(U, np.diag(1 / np.sqrt(S + epsilon)))
    components = np.dot(tmp, U.T)
    return np.dot(norm_inst, components.T)


class KMeansFeatureTransformer(object):
    def __init__(self, patches, k=1500, model_path=None):
        self.mean_distances = None
        if model_path is None:
            self.k_means = MiniBatchKMeans(n_clusters=k, compute_labels=False,
                                           reassignment_ratio=0, max_no_improvement=10, batch_size=10000,
                                           verbose=2)
            self.k_means.fit(patches)
            # update mean distances
            self.compute_mean_distances(patches)
        else:
            self.load(model_path)

    def transform(self, patches):
        return self.k_means.transform(patches)

    def predict(self, patches):
        return self.k_means.predict(patches)

    def compute_mean_distances(self, patches):
        self.mean_distances = np.mean(self.k_means.transform(patches), axis=0)

    def save(self, file_path='model/k_means_model'):
        joblib.dump(self.k_means, file_path)

    def load(self, file_path):
        self.k_means = joblib.load(file_path)


if __name__ == '__main__':
    train_y, train_x = load_pickled_dataset('data/train.pkl', whitening=False)
    test_y, test_x = load_pickled_dataset('data/test.pkl', whitening=False)

    print "STATUS: data loading done."

    # concatenate patches
    patches = np.concatenate((extract_patches(train_x), extract_patches(test_x)))

    k_means = KMeansFeatureTransformer(None, model_path='model/k_means_model_full')
    k_means.compute_mean_distances(patches)

    train_x_prime = feat_transform(train_x, k_means)
    test_x_prime = feat_transform(test_x, k_means)
