from strategies.RandomForest import RandomForest
from strategies.UniqueValues import UniqueValues
from strategies.XGBoost import XGBoost
from strategies.MissingValues import MissingValues
from strategies.MatrixCorrelation import MatrixCorrelation
from strategies.ShapleyValue import ShapleyValue

from strategies.FullPipeline import FullPipeline


import pandas as pd

df = pd.read_csv("test.csv")

#uv = UniqueValues()

label="extended_mark"

pipe = FullPipeline(steps=[
    MissingValues(),
    UniqueValues(),
    MatrixCorrelation(),
    ShapleyValue(),
    XGBoost(),
    RandomForest(),
])

pipe.run(df, label)

print(pipe.get_best_features(20))