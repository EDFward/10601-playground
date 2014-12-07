import cPickle
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from util import load_pickled_dataset


__author__ = 'junjiah'

PIC_LENGTH = 32
CHANNEL_LENGTH = 1024


def extract_patches(instances, patch_size=8):
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


def patches_vote(patch_labels, voter_num=16):
    patch_aggregate = patch_labels.reshape([len(patch_labels) / voter_num], -1)
    return [np.argmax(np.bincount(i)) for i in patch_aggregate]


class KMeansFeatureTransformer(object):
    def __init__(self, patches, k=1000, sample_size=20000, model_path=None):
        if model_path is None:
            self.k_means = KMeans(n_clusters=k, n_init=5)
            self.k_means.fit(patches[np.random.randint(len(patches), size=sample_size), :])
        else:
            self.load(model_path)

    def transform(self, instances):
        patches = extract_patches(instances)
        return self.k_means.transform(patches)

    def save(self, file_path='model/k_means_model'):
        joblib.dump(self.k_means, file_path)

    def load(self, file_path):
        self.k_means = joblib.load(file_path)


if __name__ == '__main__':

    train_y, train_x = load_pickled_dataset('data/train.pkl', whitening=True)
    test_y, test_x = load_pickled_dataset('data/test.pkl', whitening=True)

    print "STATUS: data loading done. begin loading K-means model"

    if not os.path.isfile('model/k_means_model'):
        instances = np.concatenate([train_x, test_x])
        patches = extract_patches(instances, patch_size=8)

        K = KMeansFeatureTransformer(patches, k=1000, sample_size=20000)
        K.save()
    else:
        K = KMeansFeatureTransformer(None, model_path='model/k_means_model')

    print "STATUS: K-means loading done. begin preprocessing"

    # preprocess the data
    train_patch, test_patch = K.transform(train_x), K.transform(test_x)

    dup_num = len(train_patch) / len(train_x)
    train_patch_label = np.asarray([j for i in train_y for j in [i] * dup_num])
    test_patch_label = np.asarray([j for i in test_y for j in [i] * dup_num])
    with open('data/train-patch.pkl', 'wb') as f:
        cPickle.dump(train_patch, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(train_patch_label, f, protocol=cPickle.HIGHEST_PROTOCOL)
    with open('data/test-patch.pkl', 'wb') as f:
        cPickle.dump(test_patch, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_patch_label, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print "STATUS: transformed patch data dumping finished. enjoy :)"