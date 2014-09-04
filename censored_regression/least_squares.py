import logging 

import numpy as np 

from base_estimator import BaseEstimator 

class CensoredLeastSquares(BaseEstimator):

    def __init__(self, *args, **kwargs):
        """
        Unregularized ordinary least squares with censored labels. 
        """

        BaseEstimator.__init__(self, *args, **kwargs)

    def _optimization_iteration(self, X, Y, C, eta, w):
        n_samples = X.shape[0]
        for sample_idx in xrange(n_samples):
                x = X[sample_idx]
                y = Y[sample_idx]
                predicted = np.dot(x, w)
                difference = predicted - y
                if difference > 0 and C[sample_idx]:
                    continue 
                gradient = x * difference
                w -= eta * gradient
        return w 

    def fit(self, X, Y, C):
        """
        Fit unregularized censored regression 
        
        Parameters
        -----------
        X : array, shape = (n_samples, n_features), dtype = float
            Data
        
        Y : array, shape = (n_samples,), dtype = float
            Target values 

        C : array, shape = (n_samples,), dtype = bool
            Is each sample's label censored (a lower bound), or exact? 
        """

        X, Y, C, n_samples, n_features = self._prepare_inputs(X, Y, C)
        w = np.zeros(X.shape[1]) #np.random.randn(X.shape[1]) * Y.std() / X.shape[1]
        eta = self._get_learning_rate(X,Y,C,w)
        last_empirical_error = np.inf 
        convergence_counter = 0
        for iter_idx in xrange(self.n_iters):
            w = self._optimization_iteration(X, Y, C, eta, w)
            error = self._censored_training_error(X, Y, C, w, intercept = 0)

            self.logger.info("Iter #%d, empirical error %0.4f", 
                iter_idx, 
                error, 
            )

            assert error / last_empirical_error < 1.5, "Error increasing, optimization seems to have diverged"
            if np.abs(error - last_empirical_error) <= 0.00001:
                eta /= 2.0
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter > 3:
                break 
            else:
                last_empirical_error = error 
        self.logger.info(
            "Final empirical error %0.4f", 
            self._censored_training_error(X, Y, C, w, intercept = 0),
        )
        self.coef_ = w
        return self 