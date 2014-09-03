import numpy as np 

class BaseEstimator(object):

    def __init__(
            self, 
            shuffle = True, 
            n_iters = 20, 
            eta = 0.001, 
            fit_intercept = True,
            verbose = True):
        """
        
        Parameters
        -----------
        shuffle : bool
            Randomly shuffle order of inputs before training 

        n_iters : bool
            Number of passes over the dataset 

        eta : learning rate 
        """
        self.shuffle = shuffle 
        self.n_iters = n_iters 
        self.eta = eta 
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def _prepare_inputs(self, X, Y, C):
        """
        Convert inputs to NumPy arrays and shuffle them if 
        self.shuffle is True
        """

        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
        C = np.asanyarray(C)

        assert X.ndim == 2
        (n_samples, n_features) = X.shape

        assert Y.ndim == 1
        assert C.ndim == 1

        assert len(Y) == n_samples, "Expected %d but got vector of length %d" % len(Y)
        assert len(C) == n_samples, "Expected %d but got vector of length %d" % len(C)

        if self.shuffle:
            shuffle_idx = np.arange(n_samples)
            np.random.shuffle(shuffle_idx)
            X = X[shuffle_idx]
            Y = Y[shuffle_idx]
            C = C[shuffle_idx]
        
        if self.fit_intercept:
            # add a constant column of ones to the beginning of the features
            new_shape = (n_samples, n_features + 1)
            X_old = X
            X = np.ones(new_shape, dtype = X.dtype)
            X[:, 1:] = X_old

        return X, Y, C, n_samples, n_features

    def censored_prediction_error(self, X, Y, C, w, b = 0):
        """
        Mean absolute error of predictions vs. labels (taking into account censoring)
        """
        Y_pred = np.dot(X, w) + b 
        diff = Y_pred - Y 
        overshot = (diff > 0) & C
        return np.mean(np.abs(diff[~overshot]))

    def predict(self, X):
        assert hasattr(self, "coef_"), "Must train model before making predictions"
        w = self.coef_ 
        b = self.offset_
        return np.dot(X, w) + b