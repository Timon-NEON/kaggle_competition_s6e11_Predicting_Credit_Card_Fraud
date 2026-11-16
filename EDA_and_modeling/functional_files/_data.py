import pandas as pd
from pathlib import Path

PATH_TO_DATA = '../data/'

def get_data(name='1'):

    Train = pd.read_csv(PATH_TO_DATA + f'train_{name}.csv')
    Test = pd.read_csv(PATH_TO_DATA + f'test_{name}.csv')

    # Train = Train.drop(columns=['id'])

    return Train, Test

def kaggle_solution_import(prediction: pd.Series, name: str):
    Path(PATH_TO_DATA, 'solutions').mkdir(exist_ok=True)
    test_id = pd.read_csv(PATH_TO_DATA + f'test_orig.csv')['id']
    df_solution = pd.DataFrame(
        {'id': test_id,
         'is_fraud': prediction}
    )
    df_solution.to_csv(PATH_TO_DATA + f'/solutions/solution_{name}.csv', index=False)
