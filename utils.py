import torch
import os
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc


class suppress_output:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def overlap(arr1, arr2):
    bins = np.linspace(min(min(arr1), min(arr2)), max(max(arr1), max(arr2)), 50)
    hist1, _ = np.histogram(arr1, bins=bins, density=True)
    hist2, _ = np.histogram(arr2, bins=bins, density=True)
    return np.sum(np.minimum(hist1, hist2)) * np.diff(bins)[0]


def compute_eer(distances, labels, overlap_print=False):
    pos = 0.0
    neg = 0.0
    cn = 0
    cp = 0
    pos_arr = []
    neg_arr = []

    for i in range(len(labels)):
        if labels[i] == 0:
            pos += distances[i]
            cp += 1
            pos_arr.append(distances[i])
        else:
            neg += distances[i]
            cn += 1
            neg_arr.append(distances[i])


    fprs, tprs, _ = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]

    if overlap_print:
        print('Overlap:', overlap(pos_arr, neg_arr))

    return eer, pos / cp, neg / cn
