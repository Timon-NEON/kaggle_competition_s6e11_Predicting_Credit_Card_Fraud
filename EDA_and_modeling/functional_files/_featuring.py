import pandas as pd


def simple(df_orig: pd.DataFrame):
    """
    Actions:
    Validation AUC:
    """
    df = df_orig.copy()

    df = pd.get_dummies(
        df,
    )

    return df
