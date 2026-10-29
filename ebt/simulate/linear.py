from ebt.simulate.model import Model
from sklearn import linear_model as lm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def find_alpha_for_zero_features(X, y, keep_features=None, zero_features=None, start_alpha=None, positive=True, fit_intercept=False, alpha_factors=(0.75, 1.05), progressbar=True, max_alpha_attempts=200, max_iter=50000):
    """
    Finds the Lasso alpha that zeroes out exactly ``zero_features`` coefficients
    (equivalently, keeps exactly ``keep_features``).

    Parameters:
        X (ndarray): Feature matrix of shape (m, n).
        y (ndarray): Target vector of shape (m,) or matrix (m, k).
        keep_features (int, optional): Number of coefficients to keep non-zero.
        zero_features (int, optional): Number of coefficients to set to zero.
        start_alpha (float, optional): Starting alpha for the search.
        positive, fit_intercept: Passed to the Lasso solver.
        alpha_factors (tuple): Multiplicative steps (down, up) applied to alpha per attempt.
        max_alpha_attempts (int): Maximum number of attempts.
        max_iter (int): Number of iterations for the Lasso solver.

    Returns:
        float: alpha that zeroes out exactly ``zero_features`` coefficients.
    """
    assert not (keep_features is None and zero_features is None), 'must set either keep_features or zero_features'

    progress = tqdm if progressbar else lambda x: x
    n_features = X.shape[1]
    n_observations = X.shape[0]

    if keep_features is not None:
        zero_features = n_features - keep_features
        
    if type(y) == pd.Series:
        y = y.to_frame()

    # Compute alpha_max: the smallest alpha that sets all coefficients to zero
    alpha_max = np.max(np.abs(X.T @ y)) / n_observations

    alpha = start_alpha if start_alpha is not None else alpha_max / 2
        
    for _ in progress(iter(range(max_alpha_attempts))):
        lasso = lm.LassoLars(alpha=alpha, max_iter=max_iter, positive=positive, fit_intercept=fit_intercept)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lasso.fit(X, y)

        coef_matrix = lasso.coef_.reshape(n_features, -1)
        zeroed_features = np.sum(np.all(coef_matrix == 0, axis=1))
        kept_features = n_features - zeroed_features


        if zeroed_features == zero_features:
            return alpha 

        if zeroed_features > zero_features:
            if alpha < 1e-30:
                return alpha
            alpha = alpha * alpha_factors[0]
        else:
            if zeroed_features == n_features: # Can't get more
                return alpha 
            alpha = alpha * alpha_factors[1]
    
    return alpha


class LM_LassoLarsFixedFeatures(lm.LassoLars):
    def __init__(self, alpha=1, n_features=10, progressbar=False, fit_intercept=False, verbose=False, max_iter=500, positive=True):
        self.alpha = alpha
        self.n_features = n_features
        self.progressbar = progressbar
        self.max_iter = max_iter
        super().__init__(alpha, fit_intercept=fit_intercept, verbose=verbose, max_iter=max_iter, positive=positive)

    def fit(self, X, y, Xy=None):
        self.alpha = find_alpha_for_zero_features(X, y, start_alpha=self.alpha, keep_features=self.n_features, positive=self.positive, max_iter=self.max_iter, progressbar=self.progressbar)
        return super().fit(X, y, Xy)
    
    def get_params(self, deep=True):
        return super().get_params(deep=deep)


class LinearRegression(Model):
    def __init__(self, model_or_constructor, save_betas=False, **model_hyperparameters):
        '''
        Sklearn wrapper, model_or_constructor should be either a constructor from 
            sklearn.linear_model or a instantiated object implementing the same interface.
        '''
        self.model = model_or_constructor(**model_hyperparameters) if callable(model_or_constructor) else model_or_constructor
        self.save_betas = save_betas
        self.betas = {}
        
    def fit(self, x, y, *args, **kwargs):
        x, y = x.align(y.dropna(how='all'), join='inner', axis=0)
        self.x_columns = x.columns
        self.y_columns = y.columns
        self.model.fit(x, y)
        if self.save_betas:
            self.betas[x.index[-1]] = pd.DataFrame(self.model.coef_, index=y.columns, columns=x.columns)

        return self
    
    def predict(self, x, y):
        x_predict = x.reindex(columns=self.x_columns).fillna(0)
        return pd.DataFrame(self.model.predict(x_predict), columns=self.y_columns, index=x.index)
        
    def get_params(self, deep=True):
        return super().get_params(deep=deep)
    
    
class OLS(LinearRegression):
    def __init__(self, **model_hyperparameters):
        default_parameters = {'fit_intercept': False}
        default_parameters.update(model_hyperparameters)
        super().__init__(lm.LinearRegression, **default_parameters)
        

class Ridge(LinearRegression):
    def __init__(self, **model_hyperparameters):
        default_parameters = {'fit_intercept': False}
        default_parameters.update(model_hyperparameters)
        super().__init__(lm.Ridge, **default_parameters)


class LassoLars(LinearRegression):
    def __init__(self, **model_hyperparameters):
        default_parameters = {'fit_intercept': False}
        default_parameters.update(model_hyperparameters)
        super().__init__(lm.LassoLars, **default_parameters)


class LassoLarsFixedFeatures(LinearRegression):
    def __init__(self, n_features=10, **model_hyperparameters):
        default_parameters = {'n_features': n_features, 'fit_intercept': False}
        default_parameters.update(model_hyperparameters)
        super().__init__(LM_LassoLarsFixedFeatures, **default_parameters)


class LassoCV(LinearRegression):
    def __init__(self, **model_hyperparameters):
        default_parameters = {'fit_intercept': False}
        default_parameters.update(model_hyperparameters)
        super().__init__(lm.LassoCV, **default_parameters)


class SparseMarkowitz(LinearRegression):
    def __init__(self, model_or_constructor=None, scale_factor=1, sum_to_one=True, save_betas=False, apply_default_arguments=True, **model_hyperparameters):
        '''
        Performs portfolio optimization on x by finding the solution which best achieves a constant 
        and positive target_return each day. Defaults to using a Lasso regression in order to use as
        few markets as possible.
        
        Example
            np.random.seed(0)
            ret = pd.DataFrame({'a': np.random.normal(size=(1000))}) \
                .assign(b=lambda x: -0.5 * x['a'] + 0.5 * np.random.normal(size=(1000))) \
                .assign(c=lambda x: np.random.normal(size=(1000))) \

            model = SparseMarkowitz(lm.LassoLars(fit_intercept=False, normalize=False, positive=False, alpha=0), sum_to_one=False)
            model.fit(ret, pd.Series(1, index=ret.index).to_frame())
            weights = model.predict(ret, pd.Series(1, index=ret.index).to_frame())
            ret.mean(axis=1).pipe(sharpe), (ret * weights).mean(axis=1).pipe(sharpe)
        '''
        self.scale_factor = scale_factor
        self.sum_to_one = sum_to_one
        default_parameters = {}
        if apply_default_arguments:
            default_parameters = {'fit_intercept': False, 'positive': True}

        default_parameters.update(model_hyperparameters)
        if model_or_constructor:
            self.model = model_or_constructor(**default_parameters) if callable(model_or_constructor) else model_or_constructor
        else:
            self.model = lm.LassoLars(**default_parameters)
        
    def fit(self, x, y, *args, **kwargs):
        x, y = x.align(y.dropna(how='all'), join='inner', axis=0)
        self.x_columns = x.columns
        self.model.fit(x, y.sum(axis=1) * self.scale_factor)
        return self
    
    def predict(self, x, y):
        weights = pd.Series(self.model.coef_, index=self.x_columns).rename(x.index[-1])
        if self.sum_to_one:
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = pd.Series(np.nan / len(self.x_columns), index=self.x_columns).rename(x.index[-1])
        return pd.DataFrame(weights).T.reindex(x.index).ffill().bfill()
    

class SparseMarkowitzCV(SparseMarkowitz):
    '''
    Version of SparseMarkowitz which applies cross validation using gridsearch. Requires a specific way of calling it.

    For example:
        m = ebt.simulate.linear.SparseMarkowitzCV(
            model_or_constructor=GridSearchCV, 
            apply_default_arguments=False, 
            estimator=lm.Ridge(fit_intercept=False, positive=True), 
            param_grid={'alpha': np.logspace(start=1, stop=20, base=2, num=15)},
            scoring='neg_mean_squared_error',
            cv=5,
        )
        r = ebt.predict(pnls.fillna(0), pd.DataFrame().reindex_like(pnls).fillna(1), models=[m], prediction_horizon=2,)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_params = {}

    def predict(self, x, y):
        weights = pd.Series(self.model.best_estimator_.coef_, index=self.x_columns).rename(x.index[-1])
        if self.sum_to_one:
            weights = weights / weights.sum()
            
        return pd.DataFrame(weights).T.reindex(x.index).ffill().bfill()
    
    def fit(self, x, y, *args, **kwargs):
        x, y = x.align(y.dropna(how='all'), join='inner', axis=0)

        self.x_columns = x.columns
        self.model.fit(x, y.sum(axis=1) * self.scale_factor)
        self.best_params[x.index[-1]] = self.model.best_params_
        return self
