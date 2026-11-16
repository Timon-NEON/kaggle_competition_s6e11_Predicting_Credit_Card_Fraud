import pandas as pd
import warnings
import optuna
from sklearn.feature_extraction import DictVectorizer
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pickle

print('Model Impex File: Libraries loaded!')

PATH_TO_MODELS = 'models/'

def dv_initializing(df: pd.DataFrame):
    dv = DictVectorizer(sparse=False)
    dv.fit(df.to_dict(orient='records'))
    return dv

def dv_transform(dv: DictVectorizer, df: pd.DataFrame):
    return dv.transform(df.to_dict(orient='records'))

def export_model(name: str, dv, model):
    output_file = PATH_TO_MODELS + f"{name}.bin"
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

def import_model(name: str):
    input_file = PATH_TO_MODELS + f"{name}.bin"
    with open(input_file, "rb") as f_in:
        dv, model =pickle.load(f_in)
    return dv, model