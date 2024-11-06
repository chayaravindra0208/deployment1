import pandas as pd
import json
import pickle
import streamlit as st
from datetime import timezone 
import datetime 

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters


ID_COLUMN_NAME = "ID"
JSON_FILE_NAME = "columns.json"
SCALER_MODEL_PATH = "scaling_model"
PCA_MODEL_PATH = "pca_model"
TRAIN_MODEL_PATH = "best_trained_model"

@st.cache_resource
def load_json():
    # load the json
    with open(JSON_FILE_NAME, "r") as j_file:
        json_data = json.load(j_file)

    return json_data


@st.cache_resource
def scaling_model():
    with open(SCALER_MODEL_PATH, "rb") as scaler_model:
        scaling_final_model = pickle.load(scaler_model)

    return scaling_final_model


@st.cache_resource
def pca_model():
    with open(PCA_MODEL_PATH, "rb") as pca_dim_model:
        pca_final_model = pickle.load(pca_dim_model)

    return pca_final_model


@st.cache_resource
def trained_model():
    with open(TRAIN_MODEL_PATH, "rb") as train_model:
        train_final_model = pickle.load(train_model)

    return train_final_model


def validate_utc_timestamp_column(df, column_name):
    try:
        # Attempt to convert the column to datetime with UTC format
        df[column_name] = pd.to_datetime(df[column_name], utc=True)
        return True
    except (ValueError, TypeError):
        return False


def extract_timeseries_features(activity_data:pd.DataFrame, timestamp_column) -> pd.DataFrame:
    # extract features
    extracted_features = extract_features(activity_data, column_id=ID_COLUMN_NAME, column_sort=timestamp_column, default_fc_parameters=EfficientFCParameters())
    # get the ids
    extracted_features["ID"] = extracted_features.index
    # reset the indexes
    extracted_features.reset_index(drop = True, inplace = True)

    return extracted_features


def timestamp_getter():
    dt = datetime.datetime.now(timezone.utc) 
    utc_time = dt.replace(tzinfo=timezone.utc) 
    utc_timestamp = utc_time.timestamp()

    return utc_timestamp