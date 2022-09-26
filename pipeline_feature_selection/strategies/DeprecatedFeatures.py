import pandas as pd
from strategies.Strategy import Strategy


class DeprecatedFeatures(Strategy):
    """Deprecated Features take out features deprecated for any reason."""
    name = 'deprecated_features'

    def __init__(self, dep_features_file_path, sorted_ascending=False):
        super().__init__()
        self.dep_features_file_path = dep_features_file_path
        self.sorted_ascending = sorted_ascending

    def run(self, df,label, save_csv = True):
        self.result = pd.DataFrame(columns=['feature', self.name])
        deprecated_list_of_features = pd.read_csv(self.dep_features_file_path)

        for feature in list(df.columns):
            value = 0.0
            if deprecated_list_of_features['deprecated_features'].isin([feature]).any():
                value = 1.0

            self.result = pd.concat([self.result, pd.DataFrame.from_dict({'feature': [feature], self.name: [value]})],ignore_index=True)

        self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
        self.sort_results()

        if save_csv:
            self.save()

    def get_result_to_value(self):
        """Take the features whose values are 0. This means that the feature is not deprecated"""
        if not self.result.empty:
            return self.result[self.result[self.name] == 0]["feature"].copy()
        return None
