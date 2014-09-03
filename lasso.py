import numpy as np 

from base_estimator import BaseEstimator 

class CensoredLasso(BaseEstimator):

    def __init__(self, *args, **kwargs):
        """
        Unregularized ordinary least squares with censored labels. 
        """
        self.regularization_weight = kwargs.pop('regularization_weight', 0.01)
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
        
        # the SGD form of Lasso consists of updates to 
        # two positive vector u,v 
        # such that w = u - v 
        u = np.zeros_like(w)
        v = np.zeros_like(w)
        #pos_mask = w>0
        #u[pos_mask] = w[pos_mask]
        #neg_mask = w<0
        #v[neg_mask] = -w[neg_mask]
        #v = np.zeros_like(w)
        #u = np.zeros_like(v)
        eta = self.eta 
        regularization_weight = self.regularization_weight

        last_empirical_error = np.inf 
        convergence_counter = 0
        for iter_idx in xrange(self.n_iters):
            for sample_idx in xrange(n_samples):
                x = X[sample_idx]
                y = Y[sample_idx]
                predicted = np.dot(x, w)
                difference = predicted - y
                if difference > 0 and C[sample_idx]:
                    u = np.maximum(0, u - eta * regularization_weight)
                    v = np.maximum(0, v - eta * regularization_weight)
                else: 
                    #gradient = x * difference
                    u = np.maximum(0, u - eta * (regularization_weight - (y-predicted)*x))
                    v = np.maximum(0, v - eta * (regularization_weight + (y-predicted)*x))
            w = u - v
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