#!/usr/bin/env python
from __future__ import print_function
import argparse
import gzip
import json
import numpy as np
import os
import re
from sklearn import metrics
import sys
# import warnings

nameFiles = {
    'A': '../data/kasteren/2010/datasets/houseA/names.json',
    'B': '../data/kasteren/2010/datasets/houseB/names.json',
    'C': '../data/kasteren/2010/datasets/houseC/names.json',
}

# warnings.filterwarnings('error')


def main():
    args = parseArgs(sys.argv)
    allNames = getNames(nameFiles)
    for path, fn in [(os.path.join(args.resultDir, fn), fn) for fn in
                     next(os.walk(args.resultDir))[2]]:
        with gzip.open(path, 'rb') as f:
            results = json.load(f)
        match = re.search(r'([ABC])', fn)
        if not match:
            print('No house letter found. Skipping {}'.format(fn))
            continue
        try:
            metrics = computeMetrics(results, allNames[match.group(1)])
            # except Warning as w:
            #     print('Warning on {}: {}'.format(fn, w))
            with gzip.open(os.path.join(args.metricDir, fn), 'w') as f:
                json.dump(metrics, f, sort_keys=True, indent=4,
                          separators=(',', ': '))
        except ValueError as e:
            print('ValueError on {}: {}'.format(fn, e))


def computeMetrics(results, names):
    # Flatten listes if necesary
    if type(results['y_pred'][0]) == list:
        results['y_pred'] = [i for sub in results['y_pred'] for i in sub]
    if type(results['y_true'][0]) == list:
        results['y_true'] = [i for sub in results['y_true'] for i in sub]
    y_pred = np.array(results['y_pred'])
    y_true = np.array(results['y_true'])
    cv_acc = np.array(results['acc'])
    numActNames = [(int(x[0]), x[1]) for x in names['activities'].items()]
    nums, actNames = zip(*sorted(numActNames, key=lambda x: x[0]))
    return {
        'activity_names': actNames,
        'confusion_matrix': confusion_matrix(y_true, y_pred, nums).tolist(),
        'cv_acc': cv_acc.tolist(),
        'cv_acc_mean': cv_acc.mean(),
        'cv_acc_std': cv_acc.std(),
        'f1_score': metrics.f1_score(y_true, y_pred, average='macro'),
        'accuracy_score': metrics.accuracy_score(y_true, y_pred),
        'precision_score': metrics.precision_score(y_true, y_pred,
                                                   average='macro'),
        'recall_score': metrics.recall_score(y_true, y_pred, average='macro'),
    }


def confusion_matrix(y_true, y_pred, nums):
    m = metrics.confusion_matrix(y_true, y_pred)
    row_sums = m.sum(axis=1).astype(float)
    m = m / row_sums.reshape(-1, 1)
    seen = set(np.unique(np.concatenate((y_true, y_pred))).tolist())
    for i in xrange(len(nums)):
        if nums[i] not in seen:
            m = np.insert(m, i, 0, axis=0)
            m = np.insert(m, i, 0, axis=1)
    return m


def getNames(namesFiles):
    n = {}
    for house, path in namesFiles.items():
        with open(path, 'rb') as f:
            n[house] = json.load(f)
    return n


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description=('Calculates metrics for results. '
                     'Written in Python 2.7.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('resultDir',
                        help='Directory containing results.')
    parser.add_argument('metricDir',
                        help='Directory where metrics will be saved.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
