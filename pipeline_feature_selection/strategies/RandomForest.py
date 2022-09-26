import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.strategies_utils import split_df
from strategies.Strategy import Strategy


class RandomForest(Strategy):
    """Random Forest run RandomForestRegressor of sklearn and returns the feature importance."""

    name = 'random_forest'

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.sorted_ascending = sorted_ascending

    def run(self, df,label, save_csv=True):

        if df.shape == (0, 0):
            self.result = pd.DataFrame(columns=['feature', self.name])
            print("Warning: Random Forest strategy - The dataframe is empty.")
        else:
            X_train, X_valid, y_train, y_valid = split_df(df, target=label)
            del X_valid
            del y_valid

            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            rf.fit(X_train, y_train)
            del y_train

            self.result['feature'] = X_train.columns
            del X_train

            self.result[self.name] = rf.feature_importances_

            self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
            self.sort_results()
            if save_csv:
                return self.save()
