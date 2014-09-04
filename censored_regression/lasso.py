import logging 
import numpy as np 
from base_estimator import BaseEstimator 

from parakeet import jit 

@jit 
def fast_lasso_sgd_iteration(X, Y, C, eta, u, v, regularization_weight):
    n_samples, n_features = X.shape
    for sample_idx in xrange(n_samples):
        x = X[sample_idx]
        y = Y[sample_idx]
        predicted = 0.0
        for i in xrange(n_features):
            predicted += x[i] * (u[i] - v[i])
        difference = y - predicted 
        if difference < 0 and C[sample_idx]:
            for i in xrange(n_features):
                u[i] = max(0, u[i] - eta * regularization_weight)
                v[i] = max(0, v[i] - eta * regularization_weight)
        else: 
            for i in xrange(n_features):
                gradient_i = difference * x[i]
                u[i] = max(0, u[i] - eta * (regularization_weight - gradient_i))
                v[i] = max(0, v[i] - eta * (regularization_weight + gradient_i))

class CensoredLasso(BaseEstimator):

    def __init__(self, *args, **kwargs):
        """
        Unregularized ordinary least squares with censored labels. 
        """
        self.regularization_weight = kwargs.pop('regularization_weight', 0.001)
        BaseEstimator.__init__(self, *args, **kwargs)

    def _get_linear_weights(self, u, v):
        return u - v

    def _optimization_iteration(self,X,Y,C,eta,u,v):
        assert X.ndim == 2 
        n_samples = X.shape[0]
        fast_lasso_sgd_iteration(X, Y, C, eta, u, v, self.regularization_weight)
        return u,v 
    


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
        
        w = np.zeros(X.shape[1], dtype = Y.dtype) #np.random.randn(X.shape[1]) * Y.std()
        
        # the SGD form of Lasso consists of updates to 
        # two positive vector u,v 
        # such that w = u - v 
        u = np.zeros_like(w)
        v = np.zeros_like(w)
        eta = self._get_learning_rate(X, Y, C, u, v)


        last_empirical_error = np.inf 
        n_drops = 0
        for iter_idx in xrange(self.n_iters):
            u,v = self._optimization_iteration(X,Y,C,eta,u,v)
            w = u - v
            error = self._censored_training_error(X, Y, C, w, intercept = 0)

            self.logger.info("Iter #%d, empirical error %0.4f",
                iter_idx, 
                error, 
            )
            error_ratio = error / last_empirical_error
            if np.isinf(error_ratio) or np.isnan(error_ratio) or error_ratio > 2.0:
                assert False, "Failed to converge"

            if error_ratio > 0.99999999:
                eta /= 2.0 
                self.logger.info("Dropped learning rate to %f", eta)
                n_drops += 1
            if eta < 10 ** -8 or n_drops > 5:
                break
            last_empirical_error = error 
        self.logger.info("Final empirical error %0.4f", 
            self._censored_training_error(X, Y, C, w, intercept = 0),
        )
        self.coef_ = w
       
        return self 