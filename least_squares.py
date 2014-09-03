import numpy as np 

from base_estimator import BaseEstimator 

class CensoredLeastSquares(BaseEstimator):

    def __init__(self, *args, **kwargs):
        """
        Unregularized ordinary least squares with censored labels. 
        """

        BaseEstimator.__init__(self, *args, **kwargs)

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

        w = np.random.randn(X.shape[1]) * Y.std()

        eta = self.eta 
        last_empirical_error = np.inf 
        convergence_counter = 0
        for iter_idx in xrange(self.n_iters):
            for sample_idx in xrange(n_samples):
                x = X[sample_idx]
                y = Y[sample_idx]
                predicted = np.dot(x, w)
                difference = predicted - y
                if difference > 0 and C[sample_idx]:
                    continue 
                gradient = x * difference
                w -= eta * gradient
            error = self.censored_prediction_error(X, Y, C, w)

            print "Iter #%d, empirical error %0.4f" % (
                iter_idx, 
                error, 
            )

            if np.abs(error - last_empirical_error) <= 0.00001:
                eta /= 2.0
                convergence_counter += 1
            else:
                convergence_counter = 0

            if convergence_counter > 3:
                break 
            else:
                last_empirical_error = error 
        print "Final empirical error %0.4f" % (
            self.censored_prediction_error(X, Y, C, w),
        )
        self.coef_ = w
       
        return self 