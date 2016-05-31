import arff
import pandas as pd
from pyearth import Earth
from datacleaner import autoclean
import numpy as np
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pyearth import Earth
import click

from joblib import Parallel, delayed

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EarthClassifier(BaseEstimator):
    def __init__(self, **params):
        self.clf = Pipeline([
            ('earth', Earth(**params)),
            ('logistic', LogisticRegression())])
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
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

def get_datasets(filenames, task=None):
    datasets = {}
    for filename in filenames:
        logger.info('Processing {}'.format(filename))
        try:
            if filename.endswith('arff'):
                data = arff.load(open(filename))['data']
                data = pd.DataFrame(data)
            else:
                data = pd.read_csv(filename)
                logger.info(data.shape)
        except Exception as ex:
            logger.error('error reading dataset : {}, ignoring, reason : {}'.format(filename, str(ex)))
        try:
            data = autoclean(data)
        except Exception as ex:
            logger.error('error cleaning dataset : {}, ignoring, reason : {}'.format(filename, str(ex)))
        data = data.values
        X, y = data[:, 0:-1], data[:, -1]
        if X.shape[1] <= 1:
            logger.info('{} incorrect inputs shape : {}'.format(filename, X.shape))
            continue
        logger.info(X.shape)
        if task is None:
            task = guess_task(y)
        logger.info('Task for {} : {}'.format(filename, task, y))
        datasets[filename] = X, y, task
    return datasets

def guess_task(y):
    for element in set(y):
        if type(element) == str:
            break
        if int(element) == element:
            continue
        else:
            return 'regression'
    return 'classification'

def fit_model(Xtrain, ytrain, Xtest, ytest, CLS, eval_func, **params):
    start = time.time()
    model = CLS(**params)
    model.fit(Xtrain, ytrain)
    score_train = eval_func(model.predict(Xtrain), ytrain)
    delta_t = time.time() - start
    score_test = eval_func(model.predict(Xtest), ytest)
    return {'loss': score_test, 'score_train': score_train, 'score_test': score_test, 'train_time': delta_t, 'status': STATUS_OK}

def build_fit_function_kfold(X, y, CLS, n_folds=5, preprocess=lambda p:p, random_state=42, task='classification'):
    def fit_model_(params):

        params = preprocess(params)
        if task == 'classification':
            skf = StratifiedKFold(y, n_folds, random_state=random_state)
            eval_func = lambda y_pred, y: float((y_pred != y).mean())
        else:
            skf = KFold(len(X), n_folds, random_state=random_state)
            eval_func = lambda y_pred, y: float(((y_pred - y)**2).mean())
        results = []
        for train, test in skf:
            results.append(fit_model(X[train], y[train], X[test], y[test], CLS, eval_func, **params))
        result = {}
        result['loss'] = np.mean([r['loss'] for r in results])
        result['loss_variance'] = np.var([r['loss'] for r in results])
        result['score_train'] = [r['score_train'] for r in results]
        result['score_test'] = [r['score_test'] for r in results]
        result['train_time'] = [r['train_time'] for r in results]
        result['params'] = params
        result['status'] = STATUS_OK
        return result
    return fit_model_

def propose_test_size(X, y):
    pass

@click.group()
def main():
        pass


@click.option('--pattern', default='uci/**/*.data', help='Filenames pattern for CSV datasets', required=False)
@click.option('--exclude', default='', help='Filenames pattern for excluding CSV datasets', required=False)
@click.option('--max_evals', default=10, help='Max hyperopt evaluations', required=False)
@click.option('--n_folds', default=5, help='Nb of folds', required=False)
@click.option('--random_state', default=42, help='Seed', required=False)
@click.option('--task_filter', default='none', help='classification/regression/all', required=False)
@click.option('--force_task', default=None, help='classification/regression/all', required=False)
@click.option('--nb_datasets', default=None, help='Max Nb of datasets', required=False)
@click.option('--save_results', default=True, help='Save results in DB', required=False)
@click.option('--n_jobs', default=-1, help='n_jobs', required=False)
@click.command()
def run(pattern, exclude, max_evals, n_folds, random_state, task_filter, force_task, nb_datasets, save_results, n_jobs):
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    import glob
    db = load_db()

    filenames = glob.glob(pattern)
    filenames_exclude = set(glob.glob(exclude))
    filenames = filter(lambda f:f not in filenames_exclude, filenames)

    logger.info('total number of datasets before filtering : {}'.format(len(filenames)))
    datasets = get_datasets(filenames, task=force_task)
    
    datasets = {k: (X, y, task) for k, (X, y, task) in datasets.items() if X.shape[1] > 0}
    datasets = {k: (X, y, task) for k, (X, y, task) in datasets.items() if X.shape[0] > 100}
    if task_filter != 'none':
        datasets = {k: (X, y, task) for k, (X, y, task) in datasets.items() if task == task_filter}
    datasets = datasets.items()
    if nb_datasets is not None:
        datasets = datasets[0:nb_datasets]
    logger.info('total number of datasets after filtering : {}'.format(len(datasets)))
    def hyper_optim(CLS, X, y, params, n_folds=n_folds, task='classification', preprocess=lambda p:p, random_state=random_state):
        logger.debug('X shape : {}, y shape : {}'.format(X.shape, y.shape))
        fn = build_fit_function_kfold(X, y, CLS, n_folds=n_folds, preprocess=preprocess, task=task, random_state=random_state)
        trials = Trials()
        best = fmin(fn=fn, space=params, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best, trials
    
    def get_results(datasets, model_getter, random_state=42, n_jobs=1):
        datasets_new = []
        for filename, (X, y, task) in datasets:
            CLS = model_getter[task]['cls']
            length = len(db.jobs_with(model=CLS.__name__, seed=random_state, task=task, dataset=filename))
            if length >= max_evals:
                logger.info('Skipping job on {}...already exists'.format(filename))
                continue
            else:
                datasets_new.append((filename, (X, y, task)))
        results = Parallel(n_jobs=n_jobs)(delayed(get_result)(filename, X, y, task, model_getter[task]) for filename, (X, y, task) in datasets)
        for r in results:
            for r_indiv in r:
                yield r_indiv

    def get_result(filename, X, y, task, model_getter):
        CLS = model_getter['cls']
        preprocess = model_getter.get('preprocess', lambda p:p)
        params = model_getter['params']
        try:
            best, trials = hyper_optim(CLS, X, y, params, task=task, preprocess=preprocess, random_state=random_state)
        except Exception as ex:
            logger.error('Exception : {}, ignoring.'.format(ex))
            return []
        trials_params = [trial['result']['params'] for trial in trials.trials]
        outs = []
        for result, loss, params in zip(trials.results, trials.losses(), trials_params):
            out = {'params': params, 
                   'loss': loss, 
                   'result': result, 
                   'model': CLS.__name__, 
                   'seed': random_state,
                   'task': task,
                   'dataset': filename}
            outs.append(out)
        return outs

    results = []
    logger.info('Running Earth...')
    model_getter = {
            'classification': {'cls': EarthClassifier, 'params': earth_params}, 
            'regression': {'cls': Earth, 'params': earth_params}
    }
    for result in get_results(datasets, model_getter, n_jobs=n_jobs):
        if save_results:
            print(result)
            db.safe_add_job(result, state=SUCCESS)
    model_getter = {
            'classification': {'cls': RandomForestClassifier, 'params': rf_classif_params, 'preprocess': preprocess_rf_params}, 
            'regression': {'cls': RandomForestRegressor, 'params': rf_reg_params, 'preprocess': preprocess_rf_params}
    }
    logger.info('Running RF...')
    for result in get_results(datasets, model_getter, n_jobs=1):
        if save_results:
            db.safe_add_job(result, state=SUCCESS)

@click.command()
@click.option('--out', default='out.html', help='Filename to output', required=False)
@click.option('--task_filter', default='regression', help='task', required=False)
@click.option('--dataset_pattern', default=None, help='patttern for dataset', required=False)
@click.option('--out_df', default=None, help='out dataframe csv', required=False)
def plot(out, task_filter, dataset_pattern, out_df):
    from bokeh.charts import Scatter, show, Bar
    from bokeh.io import output_file, vplot
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    import os
    import pandas as pd
    import re
    output_file(out)
    db = load_db()
    jobs = list(db.jobs_with(state=SUCCESS))
    
    if dataset_pattern is not None:
        expr = re.compile(dataset_pattern)
        def filter_data(j):
            dataset = j['content']['dataset']
            return expr.search(dataset)
        jobs = filter(filter_data, jobs)

    results = [j['content'] for j in jobs]
    assert len(results) > 0
    df = pd.DataFrame(results)
    df = df[df['task'] == task_filter]
    baseline = df['model'].iloc[0] # take any model as baseline
    df['avg_train_loss'] = df['result'].apply(lambda r:np.mean(np.sqrt(r['score_train']))) # mean in K-fold
    df['avg_test_loss'] = df['result'].apply(lambda r:np.mean(np.sqrt(r['score_test']))) # mean in K-fold
    logger.info(df['dataset'])
    df.to_csv(out_df)
    charts = []
    for loss in ('avg_train_loss', 'avg_test_loss'):
        baseline_loss_max = df[df['model'] == baseline][['dataset', loss]].groupby('dataset').agg(np.max) # max loss for the baseline model per dataset
        baseline_loss_min = df[df['model'] == baseline][['dataset', loss]].groupby('dataset').agg(np.min) # min loss for the baseline model per dataset
        def normalize(row):
            if task_filter == 'classification':
                # classification is already 'normalized'
                return row[loss]
            loss_normalized = row[loss]
            loss_normalized /= baseline_loss_max.loc[row['dataset']]
            return loss_normalized
        df[loss + '_normalized'] = df.apply(normalize, axis=1)
        df['dataset'] = df['dataset'].apply(lambda c:os.path.basename(c).replace('.data', ''))
        chart = Bar(df, label='dataset', values=loss + '_normalized', agg='min', group='model', legend='top_right', plot_width=1200, plot_height=800)
        charts.append(chart)
    fig = vplot(*charts)
    show(fig)

if __name__ == '__main__':
    main.add_command(run)
    main.add_command(plot)
    main()
