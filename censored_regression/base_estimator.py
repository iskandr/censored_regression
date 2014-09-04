import logging 
import numpy as np 

class BaseEstimator(object):

    def __init__(
            self, 
            shuffle = True, 
            n_iters = 50, 
            eta = None, 
            fit_intercept = True,
            normalize = True, 
            verbose = True):
        """
        
        Parameters
        -----------
        shuffle : bool
            Randomly shuffle order of inputs before training 

        n_iters : bool
            Number of passes over the dataset 

        eta : learning rate
            If not provided, then learned estimated from a subset
            of the data. 

        normalize : bool
            Normalize input data by subtracting its mean 
            and dividing by its variance. 

        fit_intercept : bool 
            Normalize target values by subtracting their mean and
            dividing by standard deviation of each feature. 

        verbose : bool 
            Print intermediate status updates during optimization 
        """
        self.shuffle = shuffle 
        self.n_iters = n_iters 
        self.eta = eta 
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.verbose = verbose
        if self.verbose:
            self.logger.setLevel(logging.INFO)
    
    def _get_linear_weights(self, w):
        """
        By default, assume optimization parameter is the weight vector itself. 
        Some optimization routines may decompose w into multiple vectors or
        use some alternative representation. In that case this function needs 
        to be overloaded. 
        """
        return w 

    def _estimate_intercept(self, Y):
        n = len(Y)
        Y = Y.copy()
        assert n > 0
        estimates = []
        for _ in xrange(100):
            np.random.shuffle(Y)
            Y_subset = Y[:2*n/3]
            estimates.append(Y_subset.mean())
        return np.median(estimates)
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

        assert len(Y) == n_samples, \
            "Expected %d but got vector of length %d" % (n_samples, len(Y))
        assert len(C) == n_samples, \
            "Expected %d but got vector of length %d" % (n_samples, len(C))

        if self.shuffle:
            shuffle_idx = np.arange(n_samples)
            np.random.shuffle(shuffle_idx)
            X = X[shuffle_idx]
            Y = Y[shuffle_idx]
            C = C[shuffle_idx]
        
        if self.normalize: 
            self.mean_ = X.mean(axis = 0)
            X -= self.mean_ 
            self.std_ = X.std(axis = 0)
            assert (self.std_ > 0).all(), \
                "Dimensions without variance: %s" % (self.std_ == 0).nonzero()
            X /= self.std_

        if self.fit_intercept:
            assert self.normalize, "Need data to be zero-centered"
            Ym = self._estimate_intercept(Y[~C])
            self.intercept_ = Ym
            Y -= Ym 
        else:
            self.intercept_ = 0.0 
            #X = self._extend_with_constant_column(X)
        return X, Y, C, n_samples, n_features

    def _find_best_learning_rate(
            self, X, Y, C, 
            initial_parameters, 
            subset_size = 400,
            n_epochs = 5,  
            candidate_etas = 2.0 ** -np.arange(20)):
        """
        Assumes that derived class implemented 
            1) _optimization_iteration which takes X,Y,C,eta and multiple 
                optimization parameters
            2) _get_linear_weights which takes optimization parameters to 
                generate weight vector over features 
        """
        n_samples = X.shape[0]
        subset_size = min(n_samples / 2 , subset_size)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:subset_size]
        test_idx = indices[subset_size:2*subset_size]

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        C_train = C[train_idx]

        X_test = X[test_idx]
        Y_test = Y[test_idx]
        C_test = C[test_idx]

        lowest_error = np.inf 
        best_eta = None 
        for eta in candidate_etas:
            params = [x.copy() for x in initial_parameters]
            for _ in xrange(n_epochs):
                params = \
                    self._optimization_iteration(
                        X_train, Y_train, C_train, eta, *params)
                if not isinstance(params, (list, tuple)):
                    params = [params]
            w = self._get_linear_weights(*params)
            pred = self._predict_normalized(X_test, w = w, intercept = 0)
            error = self._censored_error(pred, Y_test, C_test)
            self.logger.info(
                "Trying learning_rate = %f, error = %f", eta, error)
            if not np.isnan(error) and error < lowest_error:
                lowest_error = error
                best_eta = eta 
        assert best_eta is not None, \
            "Couldn't find a learning rate that works"
        # just to be safe, shrink the best learning rate
        # it's better to converge more slowly than risk 
        # a learning rate which worked well early but then diverges
        return best_eta / 10.0

    def _get_learning_rate(self, X, Y, C, *params):
        """
        If a learning rate wasn't set in the constructor then try to determine
        the best automatically 
        """
        if self.eta:
            return self.eta 
        else:
            return self._find_best_learning_rate(X,Y,C,params)

    def _extend_with_constant_column(self, X):
        """
        Add a constant column of ones to the beginning of the features
        """ 
        n_samples, n_features = X.shape 
        new_shape = (n_samples, n_features + 1)
        X_old = X
        X = np.ones(new_shape, dtype = X.dtype)
        X[:, 1:] = X_old
        return X

    
    def _censored_error(self, Y_pred, Y, C):
        diff = Y_pred - Y 
        overshot = (diff > 0) & C
        return np.mean(np.abs(diff[~overshot]))

    def _censored_training_error(self, X, Y, C, w = None, intercept = None):
        """
        Mean absolute error of predictions vs. labels 
        (taking into account censoring). 

        Assumes X and Y have already been normalized. 
        """
        
        if w is None:
            w = self.coef_ 
        if intercept is None:
            intercept = self.intercept_ 

        n_samples, n_features = X.shape
        assert len(Y) == n_samples, \
            "Expected len(Y) == %d but got %d" % (n_samples, len(Y))
        assert len(C) == n_samples, \
            "Expected len(C) == %d but got %d" % (n_samples, len(C))
        assert len(w) == n_features, \
            "Expected len(w) == %d but got %d" % (n_samples, len(w))


        Y_pred = np.dot(X, w) + intercept 

        return self._censored_error(Y_pred, Y, C)

    def _predict_normalized(self, X, w = None, intercept = None):
        if w is None:
            w = self.coef_ 
        if intercept is None:
            intercept = self.intercept_ 
        return np.dot(X, w) + intercept

    def predict(self, X, w = None):
        if w is None: 
            assert hasattr(self, "coef_"), \
                "Must train model before making predictions"
            w = self.coef_ 
        if self.normalize:
            X -= self.mean_ 
            X /= self.std_ 
        Y = self._predict_normalized(X, w = w)
        return Y