
import pandas as pd 

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from src.model.evaluation import print_score
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

def train_test_split(X, y, pipeline):
    # Set splitting point at 12000, since I want to spit data into 2 datasets at ratio 80:20 
    # I dont want to use random splitting since it is not practical. Since new products will be introduced in the future
    splitting_point = int(len(X)*0.8)
    X_train = pipeline.fit_transform(X[:splitting_point], y[:splitting_point])
    y_train = y[:splitting_point]
    X_validation = pipeline.transform(X[splitting_point:])
    y_validation = y[splitting_point:]
    return normalize(X_train), normalize(X_validation), y_train, y_validation

def train_model(X_train, y_train):
    sampler = SMOTE(random_state=0)
    X_rs, y_rs = sampler.fit_sample(X_train, y_train)
    lr = LogisticRegression()
    lr.fit(X_rs, y_rs)
    return lr

def performance_view():
    X, y = get_data()
    cate_0, cate_1, cate_2, products = load_items()
    pl = full_pipeline(cate_0, cate_1, cate_2, products)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, pl)
    model = train_model(X_train, y_train)
    print_score(y_true = y_validation, y_pred = model.predict(X_validation))
    
    