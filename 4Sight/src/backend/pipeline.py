from modeling import *
from data_import import *
from data_preparation import *

from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")


#!#### TO MODIFY #####!#

global db_config
db_config = {                       #Start server first
    'host': 'localhost',
    'database': 'python_db',
    'user': 'postgres',
    'password': 'postgres'
}

train_table_name = 'train_Bearing_Nasa'
test_table_name = 'test_Bearing_Nasa'
dow, month = None, None                                                             #! Get from UI
cat = None                                                                          #! Get from UI
mult_var = None                                                                     #! Get from UI
l_periods = None                                                                    #! Get from UI

#*##### Pipeline #####*#

def train_pipeline(train_table_name, db_config, dow=None, month=None, cat=None, mult_var=None, l_periods=None):
  df = read_table_from_postgres(train_table_name, db_config)
  df = data_prep(df, dow=dow, month=month, cat=cat, mult_var=mult_var, l_periods=l_periods)
  if_model = fit_IF(df)
  df_cleaned = predict_IF(if_model, df)
  fit_scaler(df_cleaned, train_table_name)
  df_scaled = apply_scaler(df_cleaned, train_table_name)

  #set_global(df_cleaned.shape[1])
  ae_fit(df_scaled, train_table_name)


def predict_pipeline(train_table_name, test_table_name):
    df_train = read_table_from_postgres(train_table_name, db_config)
    df_test = read_table_from_postgres(test_table_name, db_config)
    df_train = df_train.drop_duplicates(keep='last')
    df_test = df_test.drop_duplicates(keep='last')
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train_scaled = apply_scaler(df_train, train_table_name)
    df_test_scaled = apply_scaler(df_test, train_table_name)

    ae_path = "../models/ae_" + str(train_table_name) + ".h5"
    model = keras.models.load_model(ae_path)

    result = cluster_data(model, df_train_scaled, df_test_scaled, df_test, test_table_name)
    write_dataframe_to_postgres(result, str(test_table_name) + "_Result", db_config)