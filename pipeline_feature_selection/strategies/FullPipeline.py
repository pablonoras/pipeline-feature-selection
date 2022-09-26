import pandas as pd
from strategies.Strategy import Strategy

class FullPipeline(Strategy):

    name = "final_features_result"

    def __init__(self, steps, amount_features = 100, save_csv= True):
        super().__init__()
        self.steps = steps
        self.amount_features = amount_features
        self.save_csv = save_csv
        self.result = pd.DataFrame({"feature": []})

    def run(self, df, label):
        df_tr = df

        self.result["feature"] = df.columns
        ascending = []

        for strategy in self.steps:
            print(f"The {strategy.name} strategy is running ..")
            strategy.run(df_tr, label)
            self.result = self.result.merge(strategy.result, how="left", on="feature")
            ascending.append(strategy.sorted_ascending)

        self.result.drop(self.result[self.result['feature'] == label].index, inplace=True)
        self.result.sort_values(by=self.result.columns[1:].to_list(), ascending=ascending, inplace=True)
        self.result

        print(f"The feature selection has finished..")
        if self.save_csv:
            self.save()

    def get_best_features(self, num_features):
        return self.result["feature"][:num_features]