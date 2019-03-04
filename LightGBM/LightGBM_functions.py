#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:00:32 2018

@author: stefan
"""
import joblib
from time import time
import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV
#from sklearn.externals.joblib.parallel import Parallel, delayed

seed = 42
if joblib.cpu_count() >= 30:
    cpu_n_jobs = 10
else:
    cpu_n_jobs = joblib.cpu_count()

verbose_level = -1

LightGBM_default_params = {'boosting_type':'gbdt', 'num_leaves':31,
                           'max_depth':-1,'learning_rate':0.1, 
                           'n_estimators':100000, 'subsample_for_bin':200000,
                           'objective':'binary','min_split_gain':0,
                           'min_child_weight':1e-3,
                           'feature_fraction':0.8, 'max_bin':255,
                           'scale_pos_weight':1, 'verbose':verbose_level,
                           'seed':seed, 'num_threads':1}
EvoSearch_default_params = {'population_size':200, 'gene_mutation_prob':0.1,
                            'gene_crossover_prob':0.5, 'tournament_size':20,
                            'generations_number':100}

def Hyper_Parameter_Tuning(DataFrame, labels, weights=None, 
                           param_dict_list = None,
                           cv_folds=5, early_stopping_rounds=100,
                           search_method : "['grid','random','evolution']" = 'random',
                           objective='binary', scoring='roc_auc',
                           boosting_type = 'gbdt'):
    """ Tunes hyper parameters for LightGBM by using sklearn's GridSearchCV or
        RandomizedSearchCV with early stopping and cross validation. 
        
        search_method : Defines what search method to be used. If 'grid',
        GridSearchCV is used. If 'random', RandomizedSearchCV is used, and an
        extra parameter, 'n_iter', must be included in each param_dict.
        If 'evolution', EvolutionaryAlgorithmSearchCV is used, and extra 
        parameters can be specified in each param_dict:
            - population_size (default: 200)
            - gene_mutation_prob (default: 0.1)
            - gene_crossover_prob (default: 0.5)
            - tournament_size (default: 20)
            - generations_number (default: 100)
        
        param_dict_list : Ordered list of dicts of parameter distributions. The
        elements of the list are dicts of params and the distributions to tune 
        over. The elements of the dicts are tuned together, so different param
        dicts are tuned seperately. It can also be used to set certain 
        parameters ('set_{param_name}={param_value}'), and to trigger tuning 
        of number of estimators ('tune_estimators'). E.g., 
        >>> param_dict_list = ['set_learning_rate=0.3', 'tune_estimators',
                {'max_depth':[3,5,7], 
                 'min_child_weight':scipy.stats.randint(low=3, high=8)},
                {'gamma':[0.0,0.1,0.2], 
                 'eta':[0.1,0.2,0.3]},
                'set_learning_rate=0.03', 'tune_estimators']
        will set learning_rate to 0.3, tune number of estimators, then 
        tune max_depth and min_child_weight together, then tune gamma and eta 
        together, then set learning_rate to 0.03 and finally tune number of 
        estimators again.
        
        Returns a LightGBM Classifier model with the best parameters determined
        by the parameter search, and a result_list that contains hyper-parameter 
        cv-results for each tuning instance invoked by param_dict_list.
    """
    extra_params = {'scoring' : scoring,
                    'n_jobs' : cpu_n_jobs,
                    'iid' : False,
                    'cv' : cv_folds,
                    'verbose' : verbose_level}
    
    LightGBM_default_params['objective'] = objective
    LightGBM_default_params['boosting_type'] = boosting_type
    
        # Define classifier model
    Model = lgbm.LGBMClassifier(**LightGBM_default_params)
    
        # List of search result dicts to return to user
    result_list = []
    
    if param_dict_list is not None:
        for i,item in enumerate(param_dict_list):
            if isinstance(item, str) and item.find("set_") == 0:
            # Item is used for setting parameter.
                    # Split the string at the "=" sign, and extract 
                    # param name and value
                s = item[len("set_"):].split("=")
                param = s[0]
                value = float(s[-1])
                    # Set param to value
                Model.set_params(**{param : value})
            
            elif isinstance(item, str) and item == 'tune_estimators':
            # Item is used to trigger tuning of number of estimators
                    # Find optimal number of estimators to use.
                Model, cv_early_stopping_obj = find_estimators(
                        Model, DataFrame, labels, weights, 
                        cv_folds, early_stopping_rounds)
                print(Model)
            
            elif isinstance(item, dict):
            # Item is a dict of param dists, which are tuned together.
                item_copy = item.copy()
                
                if search_method == 'grid':
                    print(f"GridSearch for parameters : {list(item_copy.keys())} ...\n")
                    search_result = GridSearchCV(
                            estimator=Model, param_grid=item_copy, **extra_params)
                
                elif search_method == 'random':
                    assert('n_iter' in item_copy)
                    n_iter = item_copy.pop('n_iter')
                    print(f"RandomSearch for parameters : {list(item_copy.keys())} ...\n")
                    search_result = RandomizedSearchCV(
                            estimator=Model, param_distributions=item_copy,
                            n_iter=n_iter, **extra_params)
                
                elif search_method == 'evolution':
                    evo_params = EvoSearch_default_params.copy()
                    for param in EvoSearch_default_params:
                        if param in item_copy:
                            evo_params[param] = item_copy.pop(param)
                    print(f"EvolutionarySearch for parameters : {list(item_copy.keys())} ...\n")
                    search_result = EvolutionaryAlgorithmSearchCV(
                            estimator=Model, params=item_copy, **evo_params,
                            **extra_params)
                
                    # Perform search
                search_result.fit(DataFrame, labels, sample_weight=weights)
                result_list.append(search_result.cv_results_)
                
                    # Set Model parameters to new best
                for param in item_copy:
                    Model.set_params(**{param : search_result.best_params_[param]})
                print(f"Tuning for parameters : {list(item_copy.keys())} is done.\n")

                
#        # Plot Grid Search result
##    plot_grid_search(gsearch3.cv_results_, subsample, colsample_bytree, 'Subsample', 'Colsample by tree')
##    plt.show(0)
##    plt.savefig("/groups/hep/stefan/work/HP_plots/Subsample_Colsample_by_tree.png")
##    plt.close('all')
    
    
    # Model.set_params(n_jobs=cpu_n_jobs)
    
    return Model, result_list, cv_early_stopping_obj

def find_estimators(Model, DataFrame, labels, weights, cv_folds, early_stopping_rounds):
    """ Use lightgbm's cross-validation tool to find optimal number of estimators
        for the given parameters in the model."""
    dtrain = lgbm.Dataset(DataFrame, label=labels, weight=weights)
    params = Model.get_params()
    cv_early_stopping_obj = CV_EarlyStoppingTrigger(
            early_stopping_rounds=early_stopping_rounds, 
            string='auc', maximize_score=True, method='lgbm')
    params['num_threads'] = cpu_n_jobs
    params.pop('n_jobs')
    params.pop('n_estimators')
    
    lgbm.cv(params, dtrain, num_boost_round=LightGBM_default_params['n_estimators'],
            metrics='auc',
            nfold=cv_folds, stratified=True, 
            verbose_eval=True,
            callbacks=[cv_early_stopping_obj])
    Model.set_params(n_estimators=cv_early_stopping_obj.best_iteration)
    
    return Model, cv_early_stopping_obj


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1, figsize=(80,60))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


def Train(lgbm_params, DataFrame, labels, weights=None):
    """ Train model to DataFrame using lgbm_params and labels. Each instance
        is weighted by weights array. Returns a trained model."""
    lgbm_params['num_threads'] = cpu_n_jobs
    if 'n_jobs' in lgbm_params:
        lgbm_params.pop('n_jobs')
    
    Model = lgbm.LGBMClassifier(**lgbm_params)
    Model.fit(DataFrame, y=labels, sample_weight=weights)
    
    return Model, Model.get_params()
    

def Predict(DataFrame, model, var_list):
    """ Uses a trained model to predict on DataFrame to produce 'prediction'.
        From these, a BDT score is extracted and retured."""
    # model._Booster.set_attr({'num_threads' : cpu_n_jobs})
    prediction = model.predict_proba(DataFrame.loc[:, var_list])
    score = np.array([score_1 for score_0,score_1 in prediction])
    
    return score


class CV_EarlyStoppingTrigger:
    """
        Applies early stopping to xgb and lgbm cv module with 
        specified evaluation function. 
        Does not cut off last values as the usual early stopping version does.
    """
    def  __init__(self, early_stopping_rounds, string='auc', maximize_score=True, 
                  method='xgb'):
        """
        :int early_stopping_rounds: Number of rounds to use for early stopping.
        :str string: Name of the evaluation function to apply early stopping to.
        :bool maximize_score: If True, higher metric scores treated as better.
        """
        
        self.early_stopping_rounds = early_stopping_rounds
        self.string = string
        self.method = method
        self.maximize_score = maximize_score
        self.timer = []
        
        self.reset_class()
        
        
    def reset_class(self):
        
        self.best_score = None
        self.best_iteration = 0
        self.iteration = 0
        self.do_reset_class = False
        self.timer = []


    def __call__(self, callback_env):
        
        if self.do_reset_class:
            self.reset_class()
        
        self.timer.append(time())
        
        evaluation_result_list = callback_env.evaluation_result_list
        print(evaluation_result_list)
        
        if self.method.lower() == 'xgb':
            names = [self.string, 'test']
                # Test score for specific name:
            score = [x[1] for x in evaluation_result_list if 
                      all(y.lower() in x[0].lower() for y in names)][0]
        elif self.method.lower() == 'lgbm':
            names = [self.string]
                # Test score for specific name:
            score = [x[2] for x in evaluation_result_list if 
                      all(y.lower() in x[1].lower() for y in names)][0]
        
            # If first run
        if self.best_score is None:
            self.best_score = score
        
            # If better score than previous, update score
        if (self.maximize_score and score > self.best_score) or \
            (not self.maximize_score and score < self.best_score):
                self.best_iteration = self.iteration
                self.best_score = score
        
            # Trigger EarlyStoppingException from callbacks library
        elif self.iteration - self.best_iteration >= self.early_stopping_rounds:

            self.timer = np.array(self.timer[1:]) - np.array(self.timer[:-1])
            self.do_reset_class = True
            
            if self.method.lower() == 'xgb':
                from xgboost.callback import EarlyStopException
                raise EarlyStopException(self.iteration)
            
            elif self.method.lower() == 'lgbm':
                from lightgbm.callback import EarlyStopException
                raise EarlyStopException(self.iteration, self.best_score)
            
        self.iteration += 1
        
        
        
## Doesn't work with XGBoost
#def Parallel_Predict(DataFrame, model, var_list, method='predict_proba', batches_per_job=3):
#    n_batches = batches_per_job * cpu_n_jobs
#    n_samples = len(DataFrame)
#    batch_size = int(np.ceil(n_samples / n_batches))
#    parallel = Parallel(n_jobs = cpu_n_jobs)
#    results = parallel(delayed(_predict)(DataFrame, model, var_list, method, i, i + batch_size)
#                        for i in range(0, n_samples, batch_size))
#        
#    predictions = np.concatenate(results)
#    score = np.array([score_1 for score_0,score_1 in predictions])
#
#    return score
#
#def _predict(DataFrame, model, var_list, method, start, stop):
#        return getattr(model, method)(DataFrame.loc[start:stop, var_list])