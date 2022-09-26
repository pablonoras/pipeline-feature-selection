import pandas as pd
from strategies.Strategy import Strategy


class MissingValues(Strategy):
    """Missing values is a correlation with zeros."""
    name = "missing_value"

    def __init__(self, sorted_ascending=True):
        super().__init__()
        self.result = pd.DataFrame(columns=['feature', self.name])
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv=True):
        row_qty = df.shape[0]
        for feature in list(df):
            value = 1 - float(df[feature].astype(bool).sum()) / float(row_qty)
            value = abs(value)
            self.result = pd.concat([self.result, pd.DataFrame.from_dict({'feature': [feature], self.name: [value]})],ignore_index=True)

        self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
        self.sort_results()
        if save_csv:
            return self.save()

    def get_result_to_value(self, value):
        """Take the features whose values are less or equal to treshold value."""
        if not self.result.empty:
            return self.result[self.result[self.name] <= value].copy()
        return None
