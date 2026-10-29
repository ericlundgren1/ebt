try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
except ModuleNotFoundError:
    print('Optional dependency sklearn, install to use ebt.simulate.model')
    
import pandas as pd
import uuid


class Model():
    def train(self, x, y):
        raise NotImplementedError()

    def fit(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def predict(self, x, y):
        raise NotImplementedError()
    
    def __hash__(self) -> int:
        '''
        Needed to override this as otherwise __eq__ makes no sense
        '''
        return hash(self.__repr__())
    
    def __repr__(self) -> str:
        '''
        The _internal_name is supposed to remain consistent between pickles
        '''
        if '_internal_name' not in self.__dict__:
            if '_internal_name' not in self.__dict__:
                # Use a UUID to ensure a unique, but stable identifier if not set
                self._internal_name = f'{self.__class__.__name__}-{getattr(self, "_uuid", None) or uuid.uuid4()}'
                self._uuid = self._internal_name.split('-')[-1]
        
        return self._internal_name
    
    def __eq__(self, __value: object) -> bool:
        '''
        Only looking at repr to se if a model is identical to another (default objects may not be comparable if pickled)
        '''
        return self.__repr__() == __value.__repr__()
        
    def get_state(self):
        return self.state if 'state' in self.__dict__ else {}


class TestModel(Model):
    def __init__(self):
        super().__init__()
        self.xs_fit = []
        self.ys_fit = []
        self.xs_predict = []
        self.ys_predict = []

    def fit(self, x, y):
        self.xs_fit.append(x)
        self.ys_fit.append(y)
        return self

    def predict(self, x, y):
        self.xs_predict.append(x)
        self.ys_predict.append(y)
        return x


class UnityModel(Model):
    def train(self, x, y):
        return self

    def predict(self, x, y):
        return x


class ForestModel(Model):
    def __init__(self, **regressor_args):
        self.regressor_args = regressor_args
        self.forest = None

    def fit(self, x, y):
        x, y = x.align(y.dropna(how='all'), join='inner', axis=0)
        x = x.fillna(0)
        y = y.fillna(0)
        self.x_columns = x.columns
        self.y_columns = y.columns
        y_fit = y.values.ravel() if y.shape[1] == 1 else y
        self.forest = RandomForestRegressor(**self.regressor_args).fit(x, y_fit)
        return self

    def predict(self, x, y):
        x_predict = x.reindex(columns=self.x_columns).fillna(0)
        return pd.DataFrame(self.forest.predict(x_predict), columns=self.y_columns, index=x.index)


all_models = [ForestModel(n_estimators=10)]
