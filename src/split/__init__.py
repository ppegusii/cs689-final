#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import random


def main():
    pass


def subsequences(df, minSeqLen, maxSeqLen, seed=0, randomState=None):
    '''
    Split the sequence into subsequences.

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
    seed : int
        A seed for the random number generator for retrieving idential splits.
        If randomState is present, seed is not used.
    randomState : object
        Object capturing internal state of random number generator.
        If present, seed is not used.

    Returns
    -------
    subseqs : list of pandas dataframe
        Created subsequences.
        Rows have the same consistency as the df parameter.
    lens : list of ints
        Lengths of sequences in subseqs.
    '''
    subseqs = list()
    lens = list()
    if randomState:
        random.setstate(randomState)
    else:
        random.seed(seed)
    startIdx = 0
    while df.shape[0] - startIdx >= maxSeqLen:
        # Choose seqLen such that minSeqLen <= seqLen <= maxSeqLen
        endIdx = random.randint(minSeqLen, maxSeqLen) + startIdx
        subseq = df.iloc[startIdx: endIdx, :]
        subseqs.append(subseq)
        lens.append(subseq.shape[0])
        startIdx = endIdx
    subseq = df.iloc[startIdx:, :]
    subseqs.append(subseq)
    lens.append(subseq.shape[0])
    return (
        subseqs,
        lens,
    )


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
    random.seed(seed)
    randomState = random.getstate()
    subseqs, lens = subsequences(df, minSeqLen, maxSeqLen,
                                 randomState=randomState)
    train = list()
    trainLens = list()
    test = list()
    testLens = list()
    subseqLens = zip(subseqs, lens)
    for subseq, leng in subseqLens:
        # Append a sequence of lenght seqLen to the test dataframe
        #   with probability p(test_size) otherwise append to
        #   training data frame
        if random.random() < testSize:
            test.append(subseq)
            testLens.append(leng)
        else:
            train.append(subseq)
            trainLens.append(leng)
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
