"""Hidden Markov models (HMMs) with supervised training."""

import numpy as np
from scipy.misc import logsumexp

from .base import BaseSequenceClassifier
from ._utils import atleast2d_or_csr, count_trans, safe_sparse_dot


class MultinomialHMM(BaseSequenceClassifier):
    """First-order hidden Markov model with multinomial event model.

    Parameters
    ----------
    decode : string, optional
        Decoding algorithm, either "bestfirst" or "viterbi" (default).
        Best-first decoding is also called posterior decoding in the HMM
        literature.

    alpha : float
        Lidstone (additive) smoothing parameter.
    """

    def __init__(self, decode="viterbi", alpha=.01, init_eq_any=False):  # ,
        # classes=None, init_prob=None):
        self.alpha = alpha
        self.decode = decode
        self.init_eq_any = init_eq_any
        # self.classes_ = classes
        # self.intercept_init_ = init_prob
        # self.intercept_final_ = init_prob

    def fit(self, X, y, lengths):
        """Fit HMM model to data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Feature matrix of individual samples.

        y : array-like, shape (n_samples,)
            Target labels.

        lengths : array-like of integers, shape (n_sequences,)
            Lengths of the individual sequences in X, y. The sum of these
            should be n_samples.

        Notes
        -----
        Make sure the training set (X) is one-hot encoded; if more than one
        feature in X is on, the emission probabilities will be multiplied.

        Returns
        -------
        self : MultinomialHMM
        """

        alpha = self.alpha
        if alpha <= 0:
            raise ValueError("alpha should be >0, got {0!r}".format(alpha))

        X = atleast2d_or_csr(X)
        classes, y = np.unique(y, return_inverse=True)
        # classes = unique classes sorted
        # y = indices into classes above
        # print 'y.shape: {}'.format(y.shape)
        # print 'len(classes): {}'.format(len(classes))
        # print 'classes: {}'.format(classes)
        lengths = np.asarray(lengths)
        Y = y.reshape(-1, 1) == np.arange(len(classes))
        # Y has shape sum of all sequences x # of unique classes
        # the entries in the rows of Y are false except for the
        # index of the class label
        # print 'Y.shape: {}'.format(Y.shape)
        # print 'Y: {}'.format(Y)

        end = np.cumsum(lengths)
        start = end - lengths

        if self.init_eq_any:
            init_prob = np.log(Y.sum(axis=0) + alpha)
            init_prob -= logsumexp(init_prob)
            final_prob = np.log(Y.sum(axis=0) + alpha)
            final_prob -= logsumexp(final_prob)
        else:
            init_prob = np.log(Y[start].sum(axis=0) + alpha)
            init_prob -= logsumexp(init_prob)
            # print 'len(init_prob): {}'.format(len(init_prob))
            # print 'init_prob: {}'.format(init_prob)
            final_prob = np.log(Y[start].sum(axis=0) + alpha)
            final_prob -= logsumexp(final_prob)
            # print 'len(final_prob): {}'.format(len(final_prob))
            # print 'final_prob: {}'.format(final_prob)

        feature_prob = np.log(safe_sparse_dot(Y.T, X) + alpha)
        feature_prob -= logsumexp(feature_prob, axis=0)

        trans_prob = np.log(count_trans(y, len(classes)) + alpha)
        trans_prob -= logsumexp(trans_prob, axis=0)

        self.coef_ = feature_prob
        # if not self.intercept_init_:
        self.intercept_init_ = init_prob
        # if not self.intercept_final_:
        self.intercept_final_ = final_prob
        self.intercept_trans_ = trans_prob

        # if not self.classes_:
        self.classes_ = classes

        return self
