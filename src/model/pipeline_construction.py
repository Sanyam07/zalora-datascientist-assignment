# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline, FeatureUnion
from src.model.featuring import (InteractionMatrixGeneration, LatentFeatures, FeatureSelector, DateFeatures
                                , ProductFeatures, ItemDateTimeFeatures,y_ratio_by_group, y_ratio_by_list, InitialStep
                                )
from sklearn.decomposition import TruncatedSVD

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


def MF_approach(cate_0, cate_1, cate_2, products):
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

def PCA_approach(cate_0, cate_1, cate_2, products):
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

def full_pipeline(cate_0, cate_1, cate_2, products):
    date_product_features = ['view_counts', 'is_working_day', 'is_weekend_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month', 'log_session_duration']
    item_datetime_pl = item_datetime_pipeline()
    ratio_pl = ratio_pipeline()
    mf_pl = MF_approach(cate_0, cate_1, cate_2, products)
    pca_pl = PCA_approach(cate_0, cate_1, cate_2, products)
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