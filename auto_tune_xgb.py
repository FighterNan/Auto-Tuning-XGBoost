#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Description : tune the parameters automatically
    Author      : Nan Zhou
    Date        : Mar 30, 2018
"""

from __future__ import print_function
from __future__ import division

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split

import conf
from util import load_data
from util import get_performance


LOG_FILE = conf.LOG_PATH + 'tune_xgb.log'
LOG_FORMAT = conf.LOG_FORMAT

CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000


class ParamTuner:
    def __init__(self, X_train, y_train):
        self._clf = XGBClassifier(
                        learning_rate=0.01,
                        n_estimators=1000,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        scale_pos_weight=1,
                        seed=0)
        self._dtrain = xgb.DMatrix(X_train, label=y_train)
        self._X_train = X_train
        self._y_train = y_train

    @property
    def clf(self):
        return self._clf

    def show_params(self):
        logging.info("-" * 40)
        logging.info("current params:\n" + str(self._clf.get_params()))
        logging.info("-" * 40)

    def get_param(self, name):
        return self._clf.get_params()[name]

    def set_param(self, name, value):
        self._clf.set_params(**{name: value})

    def set_params(self, params):
        self._clf.set_params(**params)

    def tune_num_boost_round(self):
        logging.info("turn num_boost_round")
        history = xgb.cv(self._clf.get_params(), dtrain=self._dtrain, num_boost_round=NUM_BOOST_ROUND,
                         nfold=CV_FOLDS, metrics='auc', early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                         show_stdv=True)
        logging.info("tail of history:\n" + str(history.tail(1)))
        logging.info("learning rate: %f, best boosting num: %d" % (self.get_param('learning_rate'), history.shape[0]))
        self.set_param('n_estimators', history.shape[0])
        self.show_params()

    def grid_search(self, param_grid):
        logging.info("grid search on %s" % param_grid.keys())
        gs = GridSearchCV(estimator=self._clf, param_grid=param_grid, scoring='roc_auc',
                          n_jobs=-1, iid=False, cv=CV_FOLDS)
        gs.fit(X=self._X_train, y=self._y_train)
        logging.info("grid_scores:\n" + '\n'.join(map(str, gs.grid_scores_)))
        logging.info("best_params: " + str(gs.best_params_))
        logging.info("best_score: " + str(gs.best_score_))
        self.set_params(gs.best_params_)
        self.show_params()
        

def main(trainset_fname, ratio, exp):
    data, label, _ = load_data(trainset_fname, ratio=ratio, exp=exp)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

    tuner = ParamTuner(X_train, y_train)

    # step 1
    logging.warning("\n\nstep 1, tune n_estimators\n")
    tuner.tune_num_boost_round()

    # step 2
    logging.warning("\n\nstep 2, tune max_depth & min_child_weight\n")
    xgb_params = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 8, 2)
    }
    tuner.grid_search(xgb_params)
    xgb_params = {
        'max_depth': map(lambda x: tuner.get_param('max_depth') + x, [-1, 0, 1]),
        'min_child_weight': map(lambda x: tuner.get_param('min_child_weight') + x, [-1, 0, 1])
    }
    tuner.grid_search(xgb_params)

    # step 3
    logging.warning("\n\nstep 3, tune gamma, then re-calibrate n_estimators\n")
    xgb_params = {
        'gamma': [i / 10 for i in range(0, 5)]
    }
    tuner.grid_search(xgb_params)
    tuner.tune_num_boost_round()

    # step 4
    logging.warning("\n\nstep 4, tune subsample & colsample_bytree\n")
    xgb_params = {
        'subsample': [i / 10 for i in range(6, 10)],
        'colsample_bytree': [i / 10 for i in range(6, 10)]
    }
    tuner.grid_search(xgb_params)
    xgb_params = {
        'subsample': map(lambda x: tuner.get_param('subsample') + x, [-.05, 0, .05]),
        'colsample_bytree': map(lambda x: tuner.get_param('colsample_bytree') + x, [-.05, 0, .05])
    }
    tuner.grid_search(xgb_params)

    # final
    logging.warning("performance of final parameters:")
    clf = tuner.clf
    clf.fit(X=X_train, y=y_train)
    get_performance(pred=clf.predict(X_test), label=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("trainset", type=str, help="file path of trainset")
    parser.add_argument("-e", "--exp", type=str, help="name of experiment", default=None)
    parser.add_argument("-d", "--downsample", type=int, help="ratio, neg-num divided by pos-num", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=LOG_FILE, format=LOG_FORMAT)

    logging.warning("\n\n-------------------- start -----------------------\n")
    logging.info(args)

    main(args.trainset, args.downsample, args.exp)

    logging.warning("\n\n--------------------  end  -----------------------\n")
    print("done! chenck the log at %s" % LOG_FILE)
