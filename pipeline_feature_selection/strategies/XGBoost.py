import pandas as pd
import xgboost as xgb
from utils.strategies_utils import split_df
from strategies.Strategy import Strategy


class XGBoost(Strategy):
    """XGBoost use an xgboost model to calculate the feature importance.
    Default parameters used in xgboost:
    max_depth: 5
    eta: 1
    colsample_bytree: .3
    scale_pos_weight: 10
    eval_metric: logloss
    objective: binary_logistic
    num_round: 120
    early_Stopping_rounds: 30
    verbose_eval: 0
    importance_type: gain'''
    """

    name = 'xgboost'

    def __init__(self, max_depth=5, eta=1,
                 colsample_bytree=.3, scale_pos_weight=10, eval_metric='logloss',
                 objective='binary:logistic', num_round=120, early_stopping_rounds=30, verbose_eval=0,
                 importance_type='gain', sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.max_depth = max_depth
        self.eta = eta
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.objective = objective
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.importance_type = importance_type
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv=True):
        if df.shape == (0, 0):
            self.result = pd.DataFrame(columns=['feature', self.name])
            print("Warning: XGBoost strategy - The dataframe is empty.")
        else:
            X_train, X_valid, y_train, y_valid = split_df(df, target=label)

            all_columns = list(X_train.columns)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            del X_train
            del y_train

            dtest = xgb.DMatrix(X_valid, label=y_valid)
            del X_valid
            del y_valid

            num_round = self.num_round

            param = {'max_depth': self.max_depth,
                     'eta': self.eta,
                     'colsample_bytree': self.colsample_bytree,
                     'scale_pos_weight': self.scale_pos_weight,
                     'eval_metric': self.eval_metric,
                     'objective': self.objective
                     }
            initial_trees = xgb.train(param, dtrain, num_round, evals=[(dtest, 'dtest')],
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      verbose_eval=self.verbose_eval)

            initial_trees.predict(dtrain)
            del dtrain

            initial_trees.predict(dtest)
            del dtest

            results = initial_trees.get_score(importance_type=self.importance_type)

            for feat in list(set(all_columns) - set(results.keys())):
                results.update({feat: 0})

            self.result['feature'] = df.columns
            self.result[self.name] = self.result['feature'].map(results)

            self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
            self.sort_results()

        if save_csv:
            self.save()