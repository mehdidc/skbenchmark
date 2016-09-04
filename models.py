from pyearth import Earth
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class RegressionClassifier(BaseEstimator):
    def __init__(self, base_estimator=RandomForestClassifier()):
        self.clf = base_estimator
        self._label_enc = LabelEncoder()
        self._one_hot_enc = OneHotEncoder(sparse=False)
 
    def fit(self, X, y):
        y = self._label_enc.fit_transform(y)
        y = self._one_hot_enc.fit_transform(y[:, np.newaxis])
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        return self.clf.transform(X)
 
    def predict(self, X):
        y = self.clf.predict(X)
        y = y.argmax(axis=1)
        y = self._label_enc.inverse_transform(y)
        return y
 

class EarthClassifier(BaseEstimator):
    def __init__(self, **params):
        self.clf = Pipeline([
                ('earth', (Earth(**params))),
                ('logistic', LogisticRegression())
            ])
 
    def fit(self, X, y):
        print(set(y))
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class EarthOneVsRestClassifier(BaseEstimator):

    def __init__(self, **params):
        self.clf = OneVsRestClassifier(EarthClassifier(**params))

    def fit(self, X, y):
        return self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)


earth_params = dict(
    max_terms=hp.quniform('max_terms', 10, 100, 1), 
    max_degree=hp.quniform('max_degree', 10, 100, 1),
    allow_missing=hp.choice('allow_missing', (True, False)),
    penalty=hp.uniform('penalty', 0, 20),
    endspan_alpha=hp.uniform('endspan_alpha', 0, 1),
    minspan_alpha=hp.uniform('minspan_alpha', 0, 1),
    thresh=hp.uniform('thresh', 0, 1),
    check_every=hp.quniform('check_every', 1, 100, 1),
    allow_linear=hp.choice('allow_linear', (True, False)),
    smooth=hp.choice('smooth', (True, False)),
    enable_pruning=hp.choice('enable_pruning', (True, False)),
)

rf_reg_params = dict(
    n_estimators=hp.quniform('n_estimators', 10, 100, 1),
    min_samples_split=hp.quniform('min_samples_split', 2, 20, 1),
    min_samples_leaf=hp.quniform('min_samples_leaf', 2, 20, 1),
    max_depth=hp.choice('max_depth', (hp.quniform('max_depth_val', 5, 50, 1), None)),
    bootstrap=hp.choice('bootstrap', (True, False))
)

rf_classif_params = rf_reg_params.copy()
rf_classif_params['criterion'] =hp.choice('criterion', ('gini', 'entropy'))


def preprocess_rf_params(p):
    keys = ['n_estimators', 'min_samples_split', 'min_samples_leaf']
    for k in keys:
        p[k] = int(p[k])
    p['n_jobs'] = -1
    return p

MODELS = {
   'random_forest': {
         'classification': {'cls': RandomForestClassifier, 'params': rf_classif_params, 'preprocess': preprocess_rf_params}, 
         'regression': {'cls': RandomForestRegressor, 'params': rf_reg_params, 'preprocess': preprocess_rf_params}
    },
   'earth': {
            'classification': {'cls': EarthOneVsRestClassifier, 'params': earth_params}, 
            'regression': {'cls': Earth, 'params': earth_params}
    }
}
