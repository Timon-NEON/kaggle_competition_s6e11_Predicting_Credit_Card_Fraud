import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

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