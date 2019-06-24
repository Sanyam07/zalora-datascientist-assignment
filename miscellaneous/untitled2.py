import pandas as pd 
from sklearn.pipeline import Pipeline, FeatureUnion
from src.model.featuring import InteractionMatrixGeneration, LatentFeatures, FeatureSelector, DateFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
import random

preprocessed_df = pd.read_csv('data/processed/training.csv')
cate_0 = joblib.load('data/processed/cate_0.jl')
cate_1 = joblib.load('data/processed/cate_1.jl')
cate_2 = joblib.load('data/processed/cate_2.jl')
products = joblib.load('data/processed/products.jl')

def PCA_approach():
    cate_0_steps = [
            ('feature_selection', FeatureSelector(features = ['cate_0'])),
                        ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_0', item_set = cate_0))
                        ]
    cate_1_steps = [('feature_selection', FeatureSelector(features = ['cate_1'])),
                        ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_1', item_set = cate_1))]
    cate_2_steps = [('feature_selection', FeatureSelector(features = ['cate_2'])),
                        ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_2', item_set = cate_2))]
    product_steps = [('feature_selection', FeatureSelector(features = ['product'])),
                        ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'product', item_set = products))]
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('cate_0_steps', Pipeline(cate_0_steps))
                                               ,('cate_1_steps', Pipeline(cate_1_steps))
                                               ,('cate_2_steps', Pipeline(cate_2_steps))
                                               ,('product_steps', Pipeline(product_steps))
                                       ]
                                       )),
    ('TruncatedSVD', TruncatedSVD(50))
            ] 
    return Pipeline(steps) 

def MF_approach():
    cate_0_steps = [('feature_selection', FeatureSelector(features = ['cate_0'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_0', item_set = cate_0)),
                    ('latent_features', LatentFeatures(item_set = cate_0, feature_num = 10))]
    cate_1_steps = [('feature_selection', FeatureSelector(features = ['cate_1'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_1', item_set = cate_1)),
                    ('latent_features', LatentFeatures(item_set = cate_1, feature_num = 10))]
    cate_2_steps = [('feature_selection', FeatureSelector(features = ['cate_2'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_2', item_set = cate_2)),
                    ('latent_features', LatentFeatures(item_set = cate_2, feature_num = 10))]
    product_steps = [('feature_selection', FeatureSelector(features = ['product'])),
                     ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'product', item_set = products)),
                     ('latent_features', LatentFeatures(item_set = products, feature_num = 10))]
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('cate_0_steps', Pipeline(cate_0_steps))
                                               ,('cate_1_steps', Pipeline(cate_1_steps))
                                               ,('cate_2_steps', Pipeline(cate_2_steps))
                                               ,('product_steps', Pipeline(product_steps))
                                       ]))
            ] 
    return Pipeline(steps) 

def Date_features():
    week_day_steps = [('feature_selection', FeatureSelector(features = ['week_day'])),
                    ('one_hot_encoder', OneHotEncoder())]
    
    hour_steps = [('feature_selection', FeatureSelector(features = ['hour'])),
                    ('one_hot_encoder', OneHotEncoder())]
    
    feature_selection = [('feature_selection', FeatureSelector(features = ['is_working_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month',
                                                                            'log_session_duration'
                                                                           ], tocsr = True))
        ]
    
    steps = [
            ('date_features', DateFeatures()),
            #('feature_selection'
             #, FeatureSelector(features = ['week_day', 'is_working_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month', 'hour', 'log_session_duration']
             #, tocsr = False)),
            ('feature_union', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('week_day_steps', Pipeline(week_day_steps))
                                               ,('hour_steps', Pipeline(hour_steps))
                                               ,('feature_selection', Pipeline(feature_selection))
                                       ]))
                    ]
    return Pipeline(steps) 

def full_pipeline():
    mf_pl = MF_approach()
    date_pl = Date_features()
    pca_pl = PCA_approach()
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('mf pipeline', mf_pl)
                                               ,('date pipeline', date_pl)
                                               ,('pca pipeline', pca_pl)
                                       ]))
            ] 
    return Pipeline(steps)
    
def downsampling(matrix, y):
    y = y.reset_index(drop = True)
    #https://chrisalbon.com/machine_learning/preprocessing_structured_data/handling_imbalanced_classes_with_downsampling/
    # Indicies of each class' observations
    i_class0 = np.where(y)[0]
    i_class1 = np.where(~y)[0]
    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    # For every observation of class 0, randomly sample from class 1 without replacement
    i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False)
    # Join together class 0's target vector with the downsampled class 1's target vector
    downsampled_y = np.hstack((y[i_class1], y[i_class0_downsampled]))
    downsampled_matrix = ss.vstack([matrix[i_class1], matrix[i_class0_downsampled]])
    return downsampled_matrix, downsampled_y

def upsampling(matrix, y):
    y = y.reset_index(drop = True)
    #https://chrisalbon.com/machine_learning/preprocessing_structured_data/handling_imbalanced_classes_with_downsampling/
    # Indicies of each class' observations
    i_class0 = np.where(y)[0]
    i_class1 = np.where(~y)[0]
    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    # For every observation of class 0, randomly sample from class 1 without replacement
    i_class1_upsampled = random.choices(i_class1, k=n_class0)
    #i_class1_upsampled = np.random.choice(i_class0, size=n_class1, replace=False)
    # Join together class 0's target vector with the downsampled class 1's target vector
    upsampled_y = np.hstack((y[i_class0], y[i_class1_upsampled]))
    upsampled_matrix = ss.vstack([matrix[i_class0], matrix[i_class1_upsampled]])
    return upsampled_matrix, upsampled_y

def score(y_true, y_pred):
    female_rate = sum(y_pred[y_true]) / sum(y_true)
    male_rate = sum(~y_pred[~y_true]) / sum(~y_true)
    return (female_rate + male_rate) / 2
    
pca_pl = PCA_approach()
pca_matrix = ss.csr_matrix(pca_pl.fit_transform(preprocessed_df))

mf_pl = MF_approach()
mf_matrix = mf_pl.fit_transform(preprocessed_df)

full_pl = full_pipeline()
full_matrix = full_pl.fit_transform(preprocessed_df)

#tSVD = TruncatedSVD(200)
#tsvd_full_matrix = tSVD.fit_transform(full_matrix)
y = preprocessed_df.is_female
from imblearn.over_sampling import RandomOverSampler, SMOTE,ADASYN
#
from sklearn.preprocessing import normalize

X = normalize(full_matrix)
X = ss.csr_matrix(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)


rfc = RandomForestClassifier()

from collections import Counter
sampler = SMOTE(random_state=0)
X_rs, y_rs = sampler.fit_sample(X_train, y_train)

#X_train, y_train = downsampling(X_train, y_train)
X_train, y_train = upsampling(X_train, y_train)

rfc.fit(X_rs, y_rs)
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred, average = 'macro')
score(y_test, y_pred)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_rs, y_rs)
y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred, average = 'macro')
score(y_test, y_pred)




import numpy as np
def custom_scorer(y_true, y_pred, actual_scorer):
    score = np.nan
    try:
      score = actual_scorer(y_true, y_pred)
    except Exception: 
      pass

    return score


from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
acc = make_scorer(custom_scorer, actual_scorer = score)

random_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [5, 10, 15, 20],
 'n_estimators': [20, 40, 60, 80]}

from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(rfc, param_distributions = random_grid, n_iter=20, n_jobs= -1, cv = 5, verbose=5)
rscv.fit(X_rs, y_rs)
rscv.best_estimator_
y_pred = rscv.predict(X_test)
accuracy_score(y_test, y_pred)
score(y_test, y_pred)