import pandas as pd
from strategies.Strategy import Strategy


class UniqueValues(Strategy):
    """Unique values is where min=max."""
    name = "unique_value"

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv=True):
        for feature, type_feature in zip(list(df), df.dtypes):
            value = "not_unique"
            if df[feature].count() > 0:
                if type_feature in ['float64', 'int64']:
                    value = "unique" if min(df[feature]) == max(df[feature]) else value
                elif type_feature in ['object']:
                    value = "unique" if len(df[feature].unique()) == 1 else value
            self.result = pd.concat(
                [self.result, pd.DataFrame.from_dict({'feature': [feature], self.name: [value]})],
                ignore_index=True)

        self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
        self.result = self.result.sort_values("unique_value", ascending=self.sorted_ascending)

        if save_csv:
            self.save()