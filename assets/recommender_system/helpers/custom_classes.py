import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.base import clone
from sklearn.naive_bayes import _BaseNB
import pandas as pd


DENSITY_ESTIMATORS = {
    'GaussianMixture': GaussianMixture,
    'BayesGaussianMixture': BayesianGaussianMixture,
}


class GenerativeMixture(_BaseNB):
    """
    Generative Bayes Classifier
    This is a meta-estimator which performs generative Bayesian classification.
    Parameters
    ----------
    density_estimator : str, class, or instance
        The density estimator to use for each class.  Options are
            'GaussianMixture' : Gaussian Mixture Model
            'BayesGaussianMixture': Bayesian Gaussian Mixture
        Alternatively, a class or class instance can be specified.  The
        instantiated class should be a sklearn estimator, and contain a
        ``score_samples`` method with semantics similar to that in
        :class:`sklearn.neighbors.KDE` or :class:`sklearn.mixture.GMM`.
    **kwargs :
        additional keyword arguments to be passed to the constructor
        specified by density_estimator.

    """

    def __init__(self, density_estimator=None, **kwargs):
        if isinstance(density_estimator, str):
            dclass = DENSITY_ESTIMATORS.get(density_estimator)
            self.density_estimator = dclass(**kwargs)
        elif isinstance(density_estimator, type):
            self.density_estimator = density_estimator(**kwargs)
        else:
            self.density_estimator = density_estimator

    def fit(self, X, y):
        # X = check_array(X)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.class_prior_ = np.array([np.float(np.sum(y == y_i)) / n_samples
                                      for y_i in self.classes_])

        self.estimators_ = [clone(self.density_estimator).fit(
            # when len(X[y == c]) == 1 fit fails
            # in that case we duplicate the row
            pd.concat([X[y == c]]*2, ignore_index=True) if len(X[y == c]) == 1 else X[y == c])
            for c in self.classes_]
        return self

    def _joint_log_likelihood(self, X):
        # X = check_array(X)
        jll = np.array([np.log(prior) + dens.score_samples(X)
                        for (prior, dens)
                        in zip(self.class_prior_,
                               self.estimators_)]).T
        return jll

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, 1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        posterior = np.exp(jll)
        posterior /= posterior.sum(1)[:, np.newaxis]
        return posterior