import pandas as pd
import networkx as nx
import random
from strategies.Strategy import Strategy


class MatrixCorrelation(Strategy):
    """Matrix correlation runs a matrix correlation and group features by groups of high correlation. The threshold is define by the value on selection_type.
    By default use pearson method."""

    name = 'matrix_correlation'

    def __init__(self, sorted_ascending=False):
        super().__init__()
        self.result = pd.DataFrame()
        self.method = 'pearson'
        self.threshold = 0.99
        self.sorted_ascending = sorted_ascending

    def run(self, df,label, save_csv = True):
        columns = list(df.columns)
        columns.remove(label)

        self.correlation_matrix = df[columns].corr(method=self.method).abs()

        self.group_result = self.get_group_of_correlated_features()

        columns_to_drop = self.group_result['columns_to_drop']

        final_columns_none_correlated = list(set(df.columns) - set(columns_to_drop))
        feature = pd.DataFrame({'feature': final_columns_none_correlated})
        importance = pd.DataFrame({self.name: list(map(lambda x: 0, final_columns_none_correlated))})

        self.result = pd.concat([feature, importance], join='outer', axis=1)
        self.sort_results()

        if save_csv:
            return self.save()

    def get_strategy_result(self, df, label):
        best_columns = self.result['feature']
        columns_to_drop = list(set(df.columns) - set(best_columns))

        if label in columns_to_drop:
            columns_to_drop.remove(label)

        return df.drop(columns_to_drop, axis=1).fillna(0)

    def get_group_of_correlated_features(self):
        m = self.correlation_matrix.copy()
        m.fillna(0, inplace=True)
        t = self.threshold
        m[m > t] = 1
        m[m <= t] = 0
        G = nx.from_numpy_matrix(m.values)

        features = list(m.columns)
        result = {'uncorrelated': [], 'columns_to_drop': []}
        i = 0

        for component in list(nx.connected_components(G)):
            if len(component) < 2:
                for node in component:
                    G.remove_node(node)
                    result['uncorrelated'] += [features[node]]
            else:
                name_component = [features[node] for node in component]
                chosen = random.choice(name_component)
                result.update({'Group ' + str(i): {'chosen': chosen, 'correlated': name_component}})
                result['columns_to_drop'] += list(set(name_component) - set([chosen]))
                i += 1

        return result
