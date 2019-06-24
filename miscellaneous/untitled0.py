import pandas as pd 
from sklearn.pipeline import Pipeline, FeatureUnion
from src.model.featuring import (InteractionMatrixGeneration, LatentFeatures, FeatureSelector, DateFeatures
                                , ProductFeatures, ItemDateTimeFeatures,y_ratio_by_group, y_ratio_by_list, InitialStep
                                )
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score


raw_df = pd.read_csv('data/raw/trainingData.csv', 
                       header = None, 
                       names = ['session_id', 'start_time', 'end_time', 'product_views'])
training_labels = pd.read_csv('data/raw/trainingLabels.csv', header = None, names = ['gender'])
y = training_labels['gender'] == 'female'

cate_0 = joblib.load('data/processed/cate_0.jl')
cate_1 = joblib.load('data/processed/cate_1.jl')
cate_2 = joblib.load('data/processed/cate_2.jl')
products = joblib.load('data/processed/products.jl')


def initial_pipeline():
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               Pipeline([('date features', DateFeatures())])
                                               ,Pipeline([('product features', ProductFeatures())])
                                       ]))] 
    return Pipeline(steps)

def item_datetime_pipeline():
    steps = [('item datetime features', ItemDateTimeFeatures()),
             ('feature selection', FeatureSelector(features=['log_view_duration'], tocsr= True)),]
    return Pipeline(steps)

def ratio_pipeline():
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('ratio by weekday', Pipeline([('week_day', y_ratio_by_group('week_day'))])),
                                               ('ratio by hour', Pipeline([('hour', y_ratio_by_group('hour'))])),
                                               ('ratio by first view cate 0', Pipeline([('first_cate_0_view', y_ratio_by_group('first_cate_0_view'))])),
                                               ('ratio by first view cate 1', Pipeline([('first_cate_1_view', y_ratio_by_group('first_cate_1_view'))])),
                                               #('ratio by first view cate 2', Pipeline([('first_cate_2_view', y_ratio_by_group('first_cate_2_view'))])),
                                               #('ratio by first view product', Pipeline([('first_product_view', y_ratio_by_group('first_product_view'))])),
                                               ('ratio by cate 0 views', Pipeline([('cate_0', y_ratio_by_list('cate_0'))])),
                                               ('ratio by cate 1 views', Pipeline([('cate_1', y_ratio_by_list('cate_1'))])),
                                               #('ratio by cate 2 views', Pipeline([('cate_2', y_ratio_by_list('cate_2'))])),
                                               #('ratio by product views', Pipeline([('product', y_ratio_by_list('product'))])),
                                       ])),
            ] 
    return Pipeline(steps)


def MF_approach():
    cate_0_steps = [('feature_selection', FeatureSelector(features = ['cate_0'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_0', item_set = cate_0)),
                    ('latent_features', LatentFeatures(item_set = cate_0, feature_num = 30))]
    cate_1_steps = [('feature_selection', FeatureSelector(features = ['cate_1'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_1', item_set = cate_1)),
                    ('latent_features', LatentFeatures(item_set = cate_1, feature_num = 30))]
    cate_2_steps = [('feature_selection', FeatureSelector(features = ['cate_2'])),
                    ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'cate_2', item_set = cate_2)),
                    ('latent_features', LatentFeatures(item_set = cate_2, feature_num = 30))]
    product_steps = [('feature_selection', FeatureSelector(features = ['product'])),
                     ('interaction_matrix_generation', InteractionMatrixGeneration(val_col = 'product', item_set = products)),
                     ('latent_features', LatentFeatures(item_set = products, feature_num = 30))]
    steps = [('features', FeatureUnion(n_jobs=1, transformer_list=
                                       [
                                               ('cate_0_steps', Pipeline(cate_0_steps))
                                               ,('cate_1_steps', Pipeline(cate_1_steps))
                                               ,('cate_2_steps', Pipeline(cate_2_steps))
                                               ,('product_steps', Pipeline(product_steps))
                                       ]))
            ] 
    return Pipeline(steps) 

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
    ('TruncatedSVD', TruncatedSVD(70))
            ] 
    return Pipeline(steps) 

def full_pipeline():
    date_product_features = ['view_counts', 'is_working_day', 'is_weekend_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month', 'log_session_duration']
    item_datetime_pl = item_datetime_pipeline()
    ratio_pl = ratio_pipeline()
    mf_pl = MF_approach()
    pca_pl = PCA_approach()
    steps = [('initial_pl', InitialStep()),
             ('features', FeatureUnion(n_jobs=1, transformer_list=
               [
                       ('date product features', Pipeline([ (('feature_selection'), FeatureSelector(date_product_features, tocsr=True))])),
                       ('item_datetime_pl', item_datetime_pl),
                       ('ratio_pl', ratio_pl),
                       ('mf_pl', mf_pl),
                       ('pca_pl', pca_pl)
               ]))
             ]
             
    return Pipeline(steps)



# split 80:20  by time
pl = full_pipeline()
X_train = pl.fit_transform(raw_df[:12000], y[:12000])
y_train = y[:12000]
X_test = pl.transform(raw_df[12000:])
y_test = y[12000:]


#X_train, X_test, y_train, y_test = train_test_split(raw_df, y)
#X_train = pl.fit_transform(X_train, y_train)
#X_test = pl.transform(X_test)

from sklearn.preprocessing import normalize
X_train = normalize(X_train)
X_test = normalize(X_test)

rfc = RandomForestClassifier()

from imblearn.over_sampling import RandomOverSampler, SMOTE,ADASYN
sampler = SMOTE(random_state=0)
X_rs, y_rs = sampler.fit_sample(X_train, y_train)


rfc.fit(X_rs, y_rs)
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred)
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


#from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
#acc = make_scorer(custom_scorer, actual_scorer = score)

random_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [5, 10, 15, 20],
 'criterion': ['gini', 'entropy'],
 'n_estimators': [20, 40, 60, 80]}

from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(rfc, param_distributions = random_grid, n_iter=20, n_jobs= -1, cv = 5, verbose=5)
rscv.fit(X_rs, y_rs)
rscv.best_estimator_
y_pred = rscv.predict(X_test)
accuracy_score(y_test, y_pred)
score(y_test, y_pred)
