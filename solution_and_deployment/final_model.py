import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import pickle

# DATA
DATA_PATH = '../data/'
MODEL_NAME = "deployment/final_model"

df = pd.read_csv(DATA_PATH + 'train_1.csv')
df_task = pd.read_csv(DATA_PATH + 'test_1.csv')

x = df.drop(columns=['is_fraud'])
y = df['is_fraud']

dv = DictVectorizer(sparse=False)
x = dv.fit_transform(x.to_dict(orient='records'))
df_task = dv.transform(df_task.to_dict(orient='records'))
dtrain = xgb.DMatrix(x, label=y, feature_names=list(dv.get_feature_names_out()))
dtask = xgb.DMatrix(df_task, feature_names=list(dv.get_feature_names_out()))

print("Data prepared successfully.")

# Model
best_eta = 0.05
best_max_depth = 6
best_mcw = 35

xgb_params = {
        'eta': best_eta,
        'max_depth': best_max_depth,
        'min_child_weight': best_mcw,

        'objective': 'binary:logistic',
        'nthread': 8,

        'seed': 42,
        'verbosity': 1,
    }
model = xgb.train(xgb_params, dtrain, num_boost_round=1000)

with open(MODEL_NAME + '.bin', "wb") as f_out:
    pickle.dump((dv, model), f_out)

print("Model trained and exported successfully.")


# Making predictions
task_id = pd.read_csv(DATA_PATH + 'test_orig.csv')['id']
prediction = model.predict(dtask)
solution = pd.DataFrame({
    'id': task_id,
    'is_fraud': prediction
})
solution.to_csv(DATA_PATH + 'solutions/solution_final.csv', index=False)

print("Solution prepared and exported successfully.")