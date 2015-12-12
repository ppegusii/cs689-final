#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import random


def main():
    pass


def trainTest(df, minSeqLen, maxSeqLen, testSize=0.7, seed=0):
    '''
    Split the sequence into training and testing sets of sequences.

    Parameters
    ----------
    df : pandas dataframe
        A dataframe containing the sequence.
        The shape of the dataframe is N x M.
        Each row consists M-1 features and ends with 1 label.
    minSeqLen : int
        Minimum sequence length.
    maxSeqLen : int
        Maximum sequence length.
    testSize : float
        In [0, 1]. This is the probable proportional size of the test set.
    seed : int
        A seed for the random number generator for retrieving idential splits.

    Returns
    -------
    trainDf : pandas dataframe
        Contains concatenated training sequences.
        Rows have the same consistency as the df parameter.
    testDf : pandas dataframe
        Contains concatenated testing sequences.
        Rows have the same consistency as the df parameter.
    trainLens : list of ints
        Lengths of training sequences in trainDf.
    testLens : list of ints
        Lengths of testing sequences in testDf.
    testSize : float
        In [0, 1]. This is the true proportional size of the test set.
        Given as a number of observations in test over total number of
        observations.
    '''
    # Create a set of train and test sequences as follows
    # Create training and testing lists to house sequences
    # Choose seqLen such that minSeqLen <= seqLen <= maxSeqLen
    # Append a sequence of lenght seqLen to the test dataframe
    #   with probability p(test_size) otherwise append to
    #   training data frame

    # Create training and testing lists to house sequences
    train = list()
    trainLens = list()
    test = list()
    testLens = list()
    random.seed(seed)
    startIdx = 0
    while df.shape[0] - startIdx >= maxSeqLen:
        # Choose seqLen such that minSeqLen <= seqLen <= maxSeqLen
        endIdx = random.randint(minSeqLen, maxSeqLen) + startIdx
        seq = df.iloc[startIdx: endIdx, :]
        # Append a sequence of lenght seqLen to the test dataframe
        #   with probability p(test_size) otherwise append to
        #   training data frame
        if random.random() < testSize:
            test.append(seq)
            testLens.append(seq.shape[0])
        else:
            train.append(seq)
            trainLens.append(seq.shape[0])
        startIdx = endIdx
    seq = df.iloc[startIdx:, :]
    train.append(seq)
    trainLens.append(seq.shape[0])
    # make the train and test dataframes
    trainDf = pd.concat(train)
    testDf = pd.concat(test)
    return (
        trainDf,
        testDf,
        trainLens,
        testLens,
        float(testDf.shape[0])/df.shape[0],
    )

if __name__ == '__main__':
    main()
