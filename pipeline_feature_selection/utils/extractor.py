import re
import numpy as np
import pandas as pd
import time
import tablestk
import logging
import tablestk.stats as stats_utils
from tablestk import DATASET_PARTITION, ServingApiUrl, OrchestratorServiceClient, get_etl_pipeline, get_dataset, \
    get_train_pipeline, get_model, NodeServiceClient
from tablestk.DTO.source_configs import SnapshotSourceConfig, BigQuerySourceConfig
from tablestk.DTO.etl_pipeline import FraudPaymentsETLPipeline, FraudPaymentsETLPipelineConfig, \
    BigQueryETLPipelineConfig, ETLPipelineFactory
from tablestk.DTO.train_pipeline import TrainPipeline, TrainPipelineConfig
from tablestk.DTO.train_service_config import AutoMLTrainServiceConfig
from tablestk.utils import load_columns_list_from_file, get_available_core_metric_list, \
    get_available_core_metric_site_list, get_anomalies_df_from_anomalies_executions_df_by_index

tablestk.utils.logger.level = logging.WARNING


class RawExtractor:

    def __init__(self, dataset_name,
                 etl_description, email, label ,dataset_date_column='creation_date'):
        self.dataset_name = dataset_name
        self.etl_description = etl_description
        self.dataset_date_column = dataset_date_column
        self.dataset_label_column = label
        self.email = email
        self.features_from_query = []
        self.features_from_snapshots = []
        self.columns = []
        self.metadata = []

    def get_features_from_snapshots(self):



    def set_features_info(self, features_from_snapshots, main_table, from_str, where_str):

        bigquery_source_config = BigQuerySourceConfig(
            select_from="""
                                            SELECT scoring_id,
                                            extract(year from main.creation_date) as year, 
                                            extract(month from main.creation_date) as month
                                            {}
                                        """.format(from_str),
            train_rows_selector=where_str
        )
        bq_cols_df = bigquery_source_config.get_available_columns_df()
        features_from_query = [c for c in bq_cols_df.loc[bq_cols_df['table_name'] != main_table]['column'] if
                               c != 'scoring_id' and not "-" in c]
        metadata_cols = bq_cols_df.loc[bq_cols_df['table_name'] == main_table]['column'].to_list()
        bq_cols_df[['table_name', 'column']].groupby('table_name').count()

        columns = list(set(features_from_query + features_from_snapshots))

        self.features_from_query = features_from_query
        self.features_from_snapshots = features_from_snapshots
        self.columns = columns
        self.metadata = metadata_cols


        print("metadata_cols: {}".format(len(metadata_cols)))
        print("features_from_query: {}".format(len(features_from_query)))
        print("features_from_snapshots: {}".format(len(features_from_snapshots)))
        print("columns: {}".format(len(columns)))

        return

    def features_detector(self, prefix, columns):

        regex = re.compile(prefix)
        filtro = [i for i in columns if regex.match(i)]

        return filtro

    def run_extraction(self, features_from_snapshots, main_table, from_str, where_str, snapshot_source_config):

        self.set_features_info(features_from_snapshots,main_table,from_str,where_str)

        #Modificar el indice de cada parte en funcion de la extraccion que se busca hacer
        extraction_feature_cols_str = ",".join(self.features_from_query)
        metadata_cols_str = ",".join(self.metadata)

        time.sleep(15)  # porque sino al cluster le da ansiedad

        bigquery_source_config = BigQuerySourceConfig(
            # SELECT and FROM clause used to execute query on BigQuery
            select_from="""
                                SELECT 
                                {},
                                {}, 
                                extract(year from creation_date) as year, 
                                extract(month from creation_date) as month
                                {}
                            """.format(metadata_cols_str, extraction_feature_cols_str, from_str),
            train_rows_selector=where_str
        )

        etl_config = FraudPaymentsETLPipelineConfig(
            # This name doesn't need to be unique. It can't be longer than 10 characters. Only uppercase and underscores.
            # If already exists, then the system will append a correlative number in order to generate a unique ID.
            dataset_name=self.dataset_name,

            # Free description.
            etl_description=self.etl_description,

            # Columns list
            dataset_columns=list(set(self.metadata + self.features_from_query + ["year", "month"])),
            # si o si deben aparecer las mismas columnas invocadas en la query (como minimo)

            # Column from which we will get year and month to generate statistics by month. This is the default value.
            # This column must be in "columns" list.
            dataset_date_column=self.dataset_date_column,

            # Column used as label for training. Extra statistics will be generated by each of this column's values.
            # This column must be in "columns" list.
            dataset_label_column=self.dataset_label_column,

            # Email to notify the pipeline ending status.
            email=self.email,

            snapshot_source_config=snapshot_source_config,
            bigquery_source_config=bigquery_source_config
        )

        etl_pipeline = ETLPipelineFactory.create_etl_pipeline(etl_config)
        etl_pipeline.run()

    def get_features_from_snapshots(self):
        