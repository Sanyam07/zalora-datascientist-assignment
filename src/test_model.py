# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.externals import joblib
from src.model.evaluation import print_score
from sklearn.preprocessing import normalize

def get_data():
    raw_df = pd.read_csv('data/test_data/testData.csv', 
                       header = None, 
                       names = ['session_id', 'start_time', 'end_time', 'product_views'])
    training_labels = pd.read_csv('data/test_data/testLabels.csv', header = None, names = ['gender'])
    y = training_labels['gender'] == 'female'
    return raw_df, y

def load_models():
    try:
        model = joblib.load('models/logistic.model')
        pipeline = joblib.load('models/pipeline.pkl')
        print("Successfully load model & pipeline")
        # Save pipeline and model
    except :
        print("Fail in generating a model. The model may not be generated yet, train a model first if that is the case")
    return model, pipeline


def test_model():
    X, y = get_data()
    model, pipeline = load_models()
    print_score(y_true = y, y_pred = model.predict(normalize(pipeline.transform(X))))