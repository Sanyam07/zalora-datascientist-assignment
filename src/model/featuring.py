from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as ss
from lightfm import LightFM
import numpy as np 
import pandas as pd

def matrix_generating(df, val_col, item_set):
    def get_item_frequency(input_list):
        output = []
        for item in item_set:
            output.append(input_list.count(item))
        return ss.csr_matrix(output)
    
    sparse_matrix = ss.vstack(df[val_col].apply(get_item_frequency))
    return sparse_matrix
    
class InteractionMatrixGeneration(BaseEstimator, TransformerMixin):
    def __init__(self, val_col, item_set):
        #self.index_col = index_col
        self.val_col = val_col
        self.item_set = item_set
    
    def fit(self, df, y = None, **fit_params):
        return self    
    
    def transform(self, df, y = None, **transform_params):
        interaction_matrix = matrix_generating(df, self.val_col, self.item_set)
        return interaction_matrix
    
class LatentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, item_set, feature_num):
        self.item_set = item_set
        self.feature_num = feature_num

    def fit(self, matrix, y = None, **fit_params):
        def item_vector_generating(interaction_matrix, item_set, feature_num):
            lfmmodel = LightFM(no_components = feature_num, loss = 'warp')
            lfmmodel.fit(ss.csr_matrix(interaction_matrix), epochs = 100, num_threads = 4)
            return lfmmodel.item_embeddings
        
        #interaction_matrix = matrix_generating(df, self.index_col, self.val_col, self.item_set)
        self.item_matrix = item_vector_generating(matrix, self.item_set, self.feature_num)
        return self
    
    def transform(self, matrix, **transform_params):   
        def matrix_multiplication(interaction_matrix, item_matrix):
            result_matrix = ss.csr_matrix.dot(interaction_matrix, item_matrix)
            result_matrix_weighted = []
            weights = [i[0] for i in matrix.sum(axis=1).tolist()]
            for row, weight in zip(result_matrix, weights):
                result_matrix_weighted.append(row/weight)
            return np.array(result_matrix_weighted)
        
        #matrix = matrix_generating(df, self.index_col, self.val_col, self.item_set)
        X = matrix_multiplication(matrix, self.item_matrix)
        return ss.csr_matrix(X)

class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y = None, **fit_params):
        return self
    
    def transform(self, input_df, y = None, **transform_params):
        df = input_df.copy()
        df.start_time = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S')
        df.end_time = pd.to_datetime(df['end_time'], format='%Y-%m-%d %H:%M:%S')
        df['week_day'] = df.start_time.apply(lambda x: x.weekday())
        #df['week_day'].str.get_dummies()
        df['is_working_day'] = df['week_day'] <= 4
        df['is_weekend_day'] = df['week_day'] > 4
        df['is_sunday'] = df['week_day'] == 6 
        
        df['is_start_of_month'] = df.start_time.apply(lambda x: x.date().day <= 5)
        df['is_end_of_month'] = df.start_time.apply(lambda x: x.date().day >=25)
        df['hour'] = df.start_time.apply(lambda x: x.hour)
        df['session_duration'] = (df.end_time - df.start_time).apply(lambda x: x.seconds)
        df['log_session_duration'] = np.log(df['session_duration'])
        ## Should i remove outliner here?
        
        # number of product views per session 
        
        cols = ['week_day', 'is_working_day', 'is_weekend_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month', 'hour', 
                'session_duration', 'log_session_duration']
        return df[cols]    

class ProductFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None, **fit_params):
        return self
    
    def transform(self, X, y = None, **transform_params):
        product_features = pd.DataFrame({"product_views": X['product_views'].apply(lambda x: [i.split('/')[:4] for i in x.split(';')])})
        product_features['view_counts'] = product_features['product_views'].apply(lambda x: len(x))
        product_features['first_cate_0_view'] = product_features['product_views'].apply(lambda x: x[0][0])
        product_features['first_cate_1_view'] = product_features['product_views'].apply(lambda x: x[0][1])
        product_features['first_cate_2_view'] = product_features['product_views'].apply(lambda x: x[0][2])
        product_features['first_product_view'] = product_features['product_views'].apply(lambda x: x[0][0])
        product_features['cate_0'] = product_features['product_views'].apply(lambda x: [i[0] for i in x])
        product_features['cate_1'] = product_features['product_views'].apply(lambda x: [i[1] for i in x])
        product_features['cate_2'] = product_features['product_views'].apply(lambda x: [i[2] for i in x])
        product_features['product'] = product_features['product_views'].apply(lambda x: [i[3] for i in x])
        return product_features

class InitialStep(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None, **fit_params):
        return self
    
    def transform(self, X, y = None, **transform_params):
        df = X.copy()
        df.start_time = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S')
        df.end_time = pd.to_datetime(df['end_time'], format='%Y-%m-%d %H:%M:%S')
        df['week_day'] = df.start_time.apply(lambda x: x.weekday())
        #df['week_day'].str.get_dummies()
        df['is_working_day'] = df['week_day'] <= 4
        df['is_weekend_day'] = df['week_day'] > 4
        df['is_sunday'] = df['week_day'] == 6 
        
        df['is_start_of_month'] = df.start_time.apply(lambda x: x.date().day <= 5)
        df['is_end_of_month'] = df.start_time.apply(lambda x: x.date().day >=25)
        df['hour'] = df.start_time.apply(lambda x: x.hour)
        df['session_duration'] = (df.end_time - df.start_time).apply(lambda x: x.seconds)
        df['log_session_duration'] = np.log(df['session_duration'])
        ## Should i remove outliner here?
        
        # number of product views per session 
        
        cols = ['week_day', 'is_working_day', 'is_weekend_day', 'is_sunday', 'is_start_of_month', 'is_end_of_month', 'hour', 
                'session_duration', 'log_session_duration']
        df =  df[cols]    
    
        product_features = pd.DataFrame({"product_views": X['product_views'].apply(lambda x: [i.split('/')[:4] for i in x.split(';')])})
        product_features['view_counts'] = product_features['product_views'].apply(lambda x: len(x))
        product_features['first_cate_0_view'] = product_features['product_views'].apply(lambda x: x[0][0])
        product_features['first_cate_1_view'] = product_features['product_views'].apply(lambda x: x[0][1])
        product_features['first_cate_2_view'] = product_features['product_views'].apply(lambda x: x[0][2])
        product_features['first_product_view'] = product_features['product_views'].apply(lambda x: x[0][0])
        product_features['cate_0'] = product_features['product_views'].apply(lambda x: [i[0] for i in x])
        product_features['cate_1'] = product_features['product_views'].apply(lambda x: [i[1] for i in x])
        product_features['cate_2'] = product_features['product_views'].apply(lambda x: [i[2] for i in x])
        product_features['product'] = product_features['product_views'].apply(lambda x: [i[3] for i in x])
        return pd.concat([df, product_features], axis=1)


class ItemDateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None, **fit_params):
        return self
    
    def transform(self, X, y = None, **transform_params):
        df = X.copy()
        # average time per product view 
        #df['view_counts'] = df['product_views_array'].apply(lambda x: len(x))
        df['log_view_duration'] = np.log(df['session_duration'] / df['view_counts'])
        #cols = ['log_view_duration']
        return df 

class y_ratio_by_group(BaseEstimator, TransformerMixin):
    def __init__(self, groupByCol, tocsr = True):
        self.groupByCol = groupByCol
        self.tocsr = tocsr
    
    def fit(self, X, y = None, **fit_params):
        df = X.copy()
        df['y'] = y
        self.mean_val = sum(y)/len(y)
        groupby_df = df.groupby(self.groupByCol).agg({'y' : lambda x: sum(x) / len(x)}).reset_index()
        self.dictionary = dict(zip(groupby_df.iloc[:,0], groupby_df.iloc[:,1]))
        return self
    
    def transform(self, X, y = None, **transform_params):
        def get_value(x):
            try:
                return self.dictionary[x]
            except:
                pass
            return self.mean_val
            #return np.nan
        ratio_col = X[self.groupByCol].apply(lambda x: get_value(x))
        if self.tocsr:
            return ss.csr_matrix(ratio_col.values.reshape(-1,1))
        return ratio_col

class y_ratio_by_list(BaseEstimator, TransformerMixin):
    def __init__(self, groupByCol, tocsr = True):
        self.groupByCol = groupByCol
        self.tocsr = tocsr
    
    def fit(self, X, y = None, **fit_params):
        
        def flattening(lists, a_list):
            assert len(lists) == len(a_list)
            flattened_lists = []
            new_a_list = []
            for inner_list, element in zip(lists, a_list):
                flattened_lists = flattened_lists + inner_list
                new_a_list = new_a_list + [element] * len(inner_list)
            return pd.DataFrame({'categorical_col': flattened_lists, 'y': new_a_list})
        
        df = flattening(X[self.groupByCol], y)
        self.mean_val = sum(y)/len(y)
        groupby_df = df.groupby('categorical_col').agg({'y' : lambda x: sum(x) / len(x)}).reset_index()
        self.dictionary = dict(zip(groupby_df.iloc[:,0], groupby_df.iloc[:,1]))
        return self
    
    def transform(self, X, y = None, **transform_params):
        def get_value(x):
            try:
                return self.dictionary[x]
            except:
                pass
            return self.mean_val
            #return np.nan
        
        ratio_col = X[self.groupByCol].apply(lambda x: np.array([get_value(i) for i in x]))
        
        if self.tocsr:
            return ss.csr_matrix(ratio_col.apply(lambda x: x.mean()).values.reshape(-1,1))
        return ratio_col.apply(lambda x: x.mean())

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features, tocsr = False):
        self.features = features
        self.tocsr = tocsr
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.tocsr:
            return ss.csr_matrix(X.loc[:, self.features].astype(np.float))
        return X.loc[:, self.features]
    