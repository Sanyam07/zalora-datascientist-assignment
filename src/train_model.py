import pandas as pd 

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

from src.model.pipeline_construction import full_pipeline

def get_data():
    raw_df = pd.read_csv('data/raw/trainingData.csv', 
                       header = None, 
                       names = ['session_id', 'start_time', 'end_time', 'product_views'])
    training_labels = pd.read_csv('data/raw/trainingLabels.csv', header = None, names = ['gender'])
    y = training_labels['gender'] == 'female'
    return raw_df, y

def load_items():
    cate_0 = joblib.load('data/processed/cate_0.jl')
    cate_1 = joblib.load('data/processed/cate_1.jl')
    cate_2 = joblib.load('data/processed/cate_2.jl')
    products = joblib.load('data/processed/products.jl')
    return cate_0, cate_1, cate_2, products

def train_model(X, y, pipeline):
    X_train = normalize(pipeline.fit_transform(X, y))
    sampler = SMOTE(random_state=0)
    X_rs, y_rs = sampler.fit_sample(X_train, y)
    lr = LogisticRegression()
    try:
        lr.fit(X_rs, y_rs)
        joblib.dump(lr, 'models/logistic.model')
        joblib.dump(pipeline, 'models/pipeline.pkl')
        print("Successfully generate a logisic model in models folder")
    # Save pipeline and model
    except :
        print("Fail in generating a model")

def model_generating():
    X, y = get_data()
    cate_0, cate_1, cate_2, products = load_items()
    pl = full_pipeline(cate_0, cate_1, cate_2, products)
    train_model(X, y, pl)
    
    
    