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

print('Data File: Libraries loaded!')

PATH_TO_DATA = '../data/'

def get_data(name='1'):

    Train = pd.read_csv(PATH_TO_DATA + f'train_{name}.csv')
    Test = pd.read_csv(PATH_TO_DATA + f'test_{name}.csv')

    # Train = Train.drop(columns=['id'])

    return Train, Test

def kaggle_solution_import(prediction: pd.Series, name: str):
    test_id = pd.read_csv(PATH_TO_DATA + f'test_orig.csv')['id']
    df_solution = pd.DataFrame(
        {'id': test_id,
         'is_fraud': prediction}
    )
    df_solution.to_csv(PATH_TO_DATA + f'/solutions/solution_{name}.csv', index=False)
