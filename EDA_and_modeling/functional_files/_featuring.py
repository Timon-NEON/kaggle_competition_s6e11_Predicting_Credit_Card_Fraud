import pandas as pd
import warnings
import optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print('Featuring File: Libraries loaded!')


def simple(df_orig: pd.DataFrame):
    """
    Actions: one_hot_encoding
    Validation AUC: 0.96354
    """
    df = df_orig.copy()

    df = pd.get_dummies(
        df,
    )

    return df
