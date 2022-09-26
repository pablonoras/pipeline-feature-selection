import pandas as pd
from strategies.Strategy import Strategy


class LabelCorrelation(Strategy):
    """Label correlation is the correlation of each feature with the "label" passed."""
    name = 'label_correlation'
    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.decision_column = 'label_correlation'
        self.sorted_ascending = sorted_ascending

    def run(self, df, label, save_csv = True):
        self.result = pd.DataFrame(columns=['feature', self.decision_column])
        for feature in list(set(df) - set(label)):
            value = df[label].fillna(0).corr(df[feature].fillna(0)).copy()
            value = abs(value)
            if pd.isna(value):
                value = 0.0
            self.result = pd.concat(
                [self.result, pd.DataFrame.from_dict({'feature': [feature], self.decision_column: [value]})],
                ignore_index=True)

        self.result.drop(self.result[self.result['feature']==label].index, inplace=True)
        self.sort_results()

        if save_csv:
            return self.save()