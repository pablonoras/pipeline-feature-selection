import pandas as pd
from strategies.Strategy import Strategy
from tablestk import get_train_pipeline, get_model, DATASET_PARTITION, EarlyStoppingMode, EarlyStoppingMonitor
from tablestk.DTO.train_pipeline import TrainPipelineConfig, TrainPipeline
from tablestk.DTO.train_service_config import AutoMLTrainServiceConfig


class AutoML(Strategy):

    name = "auto_ml"

    def __init__(self, model_name, extraction_id, objective='MAXIMIZE_AU_PRC', budget_hours=3, early_stopping=True, weight_column=None,
                 fraud_mark='recoverable_mark', date_column='creation_date',
                 model_description="Feature Importance from a autoML training",
                 email="marcos.montero@mercadolibre.com", sorted_ascending=False):
        self.objective = objective
        self.budget_hours = budget_hours
        self.early_stopping = early_stopping
        self.weight_column = weight_column
        self.model_name = model_name,
        self.model_description = model_description
        self.dataset_id = extraction_id
        self.fraud_mark = fraud_mark
        self.date_column = date_column
        self.email = email
        self.result = pd.DataFrame()
        self.sorted_ascending = sorted_ascending

    def run(self, train_columns, save_csv=True):

        automl_train_service_config = AutoMLTrainServiceConfig(
                objective=self.objective,
                budget_hours=self.budget_hours,
                early_stopping=self.early_stopping,
                weight_column=self.weight_column,
                time_column=self.date_column)

        train_pipeline_config = TrainPipelineConfig(
            model_name= self.model_name,
            model_description= self.model_description,
            dataset_id=self.dataset_id,
            train_columns=train_columns,
            label_column=self.fraud_mark,

            train_service_config=automl_train_service_config,

            predictions_include_columns=[],  # columnas de la main
            check_stats=True,
            email=self.email)

        train_pipeline = TrainPipeline(config=train_pipeline_config)
        train_pipeline.run()
        self.result = pd.DataFrame(train_pipeline.get_features_importance_sorted(), columns=['feature', self.name])
        self.sort_results()

        if save_csv:
            self.save()


    def features_cumulative_importance(self, features, threshold=0.9):
        rows = len(self.result)
        cumulative_importance = 0
        idx = 0
        for i in range(0, rows):
            cumulative_importance += features.loc[i][1]
            if cumulative_importance >= threshold:
                idx = i
                break
        importance = threshold * 100
        sorted_ = features.sort_values(['importance'], ascending=False).round(5)
        more_important = sorted_[0:idx]
        more_important.to_csv(f'features_with_{importance}P_cum_importance.csv', index=False)
        return more_important

