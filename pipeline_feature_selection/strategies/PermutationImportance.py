import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.strategies_utils import split_df
import rfpimp
from strategies.Strategy import Strategy


class PermutationImportance(Strategy):
    """Try permutation importance: randomly permute an individual column for test-set and evaluate how much performance is lost by this transformation.
    Record this diff for each feature."""

    name = 'permutation_importance'

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.sorted_ascending = sorted_ascending

    def run(self, df,label, save_csv=True):

        if df.shape == (0, 0):
            self.result = pd.DataFrame(columns=['feature', self.name])
            print("Warning: Permutation Importance strategy - The dataframe is empty.")
        else:
            print('splitting dataset')
            X_train, X_valid, y_train, y_valid = split_df(df, target=label)

            print('training regressor...')
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            rf.fit(X_train, y_train)
            del y_train

            self.result = pd.DataFrame()
            del X_train
            print('obtaining importances...')
            imp = rfpimp.importances(rf, X_valid, y_valid)  # permutation importance

            imp = imp.loc[imp['Importance'] > 1e-6]

            imp['feature'] = list(imp.index)
            imp.rename(columns={'Importance': self.name}, inplace=True)

            self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
            self.sort_results()

            if save_csv:
                self.save()
