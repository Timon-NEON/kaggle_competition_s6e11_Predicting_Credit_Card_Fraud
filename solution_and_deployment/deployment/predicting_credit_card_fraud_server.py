import pickle

import xgboost as xgb
from flask import Flask, jsonify, request


# Data exporting
MODEL_FILE = 'final_model.bin'
THRESHOLD = 0.1

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)


# Requests handling

app = Flask('card_fraud')

@app.route('/predict', methods=['POST'])
def predict():
    case = request.get_json()

    X = dv.transform(case)
    dX = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    y_predict = model.predict(dX)[0]
    is_fraud = y_predict >= THRESHOLD

    result = {
        'fraud_probability': float(y_predict),
        'is_fraud': bool(is_fraud)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
