import numpy as np
import pandas as pd
from strategies.Strategy import Strategy
from strategies.ShapleyBase import XGBoostClassifier
from utils.strategies_utils import split_df


# Parameters... Seed for the split, seed for the random variable, max_depth for the XGBoostClassifier.


class RandomUpShapley(Strategy):
    '''Similar to ShapleyValue, it adds a random variable and considers all the features that are strictly better
    than this random feature.'''

    name = 'random_up_shapley'

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.split_speed = 0
        self.random_seed = 0
        self.max_depth = 5
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv=True):
        if df.shape == (0, 0):
            self.result = pd.DataFrame([[self.name, 0.0]], columns=['feature', self.name])
            print("Warning: Random Up Shapley strategy - The dataframe is empty.")
        else:
            shapley = XGBoostClassifier(self.max_depth)

            X_train, X_valid, y_train, y_valid = split_df(df, seed=self.split_speed,
                                                          target=label)
            np.random.seed(self.random_seed)
            X_train[self.name] = np.random.randint(2, size=X_train.shape[0])

            # Random variable.

            shapley.fit(X_train, y_train)

            result = shapley.get_ranked_features()
            result.rename(columns={'importance': self.name}, inplace=True)

            self.result = result

            del X_train
            del X_valid
            del y_train
            del y_valid

            self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
            self.sort_results()
        if save_csv:
            return self.save()

        #value_of_feature_random = float(self.result[self.result['feature'] == 'shapley_random']['importance'])
        #self.result = self.result[self.result['importance'] > value_of_feature_random]