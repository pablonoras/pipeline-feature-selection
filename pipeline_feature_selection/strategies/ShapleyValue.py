from strategies.Strategy import Strategy
from strategies.ShapleyBase import XGBoostClassifier
from utils.strategies_utils import split_df
import pandas as pd


class ShapleyValue(Strategy):
    '''Shapley value runs an XGBoost model.
    The Shapley value ensures each actor gains as much or more as they would have from acting independently.
    The value obtained is critical because otherwise, there is no incentive for actors to collaborate.'''

    name = 'shapley_value'

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv=True):
        shapley = XGBoostClassifier()

        if df.shape == (0, 0):
            self.result = pd.DataFrame(columns=['feature', self.name])
            print("Warning: Shapley Value strategy - The dataframe is empty.")
        else:
            X_train, X_valid, y_train, y_valid = split_df(df, target=label)

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