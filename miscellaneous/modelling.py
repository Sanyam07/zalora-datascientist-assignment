import pandas as pd 
from scipy import sparse as ss
from lightfm import LightFM
import numpy as np 


training[['session_id', 'cate_0']]
training.set_index('session_id')['cate_0'].apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'cate_0'}).reset_index(drop = True)
## Unnest data 
#def unnest_data(df, index_col, val_col):
#    return df.set_index(index_col)[val_col].apply(pd.Series).stack().reset_index(level=0).rename(columns={0:val_col}).reset_index(drop = True)

## generate matrix 
def matrix_generating(df, index_col, val_col, item_set):
    def get_item_frequency(input_list):
        output = []
        for item in item_set:
            output.append(input_list.count(item))
        return ss.csr_matrix(output)
    
    sparse_matrix = ss.vstack(df[val_col].apply(get_item_frequency))
    return sparse_matrix
    
## generate feature vectors 
def item_vector_generating(interaction_matrix, item_set, feature_num):
    lfmmodel = LightFM(no_components = feature_num, loss = 'warp')
    lfmmodel.fit(ss.csr_matrix(interaction_matrix), epochs = 100, num_threads = 4)
    return lfmmodel.item_embeddings

def matrix_multiplication(interaction_matrix, item_matrix):
    result_matrix = ss.csr_matrix.dot(interaction_matrix, item_matrix)
    result_matrix_weighted = []
    weights = [i[0] for i in matrix.sum(axis=1).tolist()]
    for row, weight in zip(result_matrix, weights):
        result_matrix_weighted.append(row/weight)
    return np.array(result_matrix_weighted)
    
## session vectors

## predict

matrix = matrix_generating(training, index_col = 'session_id', val_col = 'cate_0', item_set=cate_0)

item_matrix = item_vector_generating(matrix,item_set=cate_0, feature_num = 10)
#X = ss.csr_matrix.dot(matrix, item_matrix)
X = matrix_multiplication(matrix, item_matrix)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


label_encoder = LabelEncoder()
cate_0_num = label_encoder.fit_transform(training['cate_0'].apply(lambda x:x[0]))
one_hot_encoder = OneHotEncoder()
X = one_hot_encoder.fit_transform(cate_0_num.reshape(-1, 1))


y = training.is_female

X_train, X_test, y_train, y_test = train_test_split(X, y)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred)

