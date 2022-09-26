from utils.strategies_utils import split_df
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from strategies.Strategy import Strategy


def imp_df(column_names, importances):
    df_ = pd.DataFrame({'feature': column_names, 'importance': importances}).sort_values('importance',ascending=False).reset_index(drop=True)
    return df_


def drop_col_feat_imp(model, X_train, y_train):
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    importances = []

    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
        importances.append(benchmark_score - drop_col_score)

    importances_df = imp_df(X_train.columns, importances)
    return importances_df


class DropFeature(Strategy):
    """Drop feature is very time expensive strategy. This strategy runs an Random Forest model, then take out a
    feature, runs random forest agin and save the diffrence between all features. The result goes from positives to
    negatives values, where negatives mean that take out that feature improve the final result.
    """

    name = "drop_feature"

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame({"feature": [], "importance": []})
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv = True):
        if df.shape[0] > 0:
            X_train, X_valid, y_train, y_valid = split_df(df, target=label)

            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            rf.fit(X_train, y_train)

            self.result = drop_col_feat_imp(rf, X_train, y_train)
            del X_train
            del X_valid
            del y_train
            del y_valid
        else:
            print("Warning: Drop Feature strategy - The dataframe is empty.")

        self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
        self.sort_results()

        if save_csv:
            self.save()


