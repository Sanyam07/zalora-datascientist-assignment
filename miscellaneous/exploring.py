import pandas as pd 
from sklearn.externals import joblib

training = pd.read_csv('data/raw/trainingData.csv', 
                       header = None, 
                       names = ['session_id', 'start_time', 'end_time', 'product_views'])
training_labels = pd.read_csv('data/raw/trainingLabels.csv', header = None, names = ['gender'])
training['is_female'] = training_labels['gender'] == 'female'

training.start_time = pd.to_datetime(training['start_time'], format='%Y-%m-%d %H:%M:%S')
training.end_time = pd.to_datetime(training['end_time'], format='%Y-%m-%d %H:%M:%S')


training['week_day'] = training.start_time.apply(lambda x: x.weekday())
training['start_of_months'] = training.start_time.apply(lambda x: x.date().day <= 5)
training['end_of_months'] = training.start_time.apply(lambda x: x.date().day >=25)
training['hour'] = training.start_time.apply(lambda x: x.hour)
training['session_duration'] = (training.end_time - training.start_time).apply(lambda x: x.seconds)

training.groupby(['week_day']).agg({'is_female': lambda x: sum(x)/sum(training['is_female'])})
training.groupby(['start_of_months']).agg({'is_female': lambda x: sum(x)/sum(training['is_female'])})
training.groupby(['end_of_months']).agg({'is_female': lambda x: sum(x)/sum(training['is_female'])})
training.groupby(['hour']).agg({'is_female': lambda x: sum(x)/sum(training['is_female'])})

training.groupby(['week_day']).agg({'is_female': lambda x: sum(~x)/sum(~training['is_female'])})
training.groupby(['start_of_months']).agg({'is_female': lambda x: sum(~x)/sum(~training['is_female'])})
training.groupby(['end_of_months']).agg({'is_female': lambda x: sum(~x)/sum(~training['is_female'])})
training.groupby(['hour']).agg({'is_female': lambda x: sum(~x)/sum(~training['is_female'])})
## count the categorical 
## Time features 

## product Views
training['product_views_array'] = training['product_views'].apply(lambda x: [i.split('/')[:4] for i in x.split(';')])
training['cate_0'] = training['product_views_array'].apply(lambda x: [i[0] for i in x])
training['cate_1'] = training['product_views_array'].apply(lambda x: [i[1] for i in x])
training['cate_2'] = training['product_views_array'].apply(lambda x: [i[2] for i in x])
training['product'] = training['product_views_array'].apply(lambda x: [i[3] for i in x])
## Categorical features 

## Item features 
list_flattening = lambda x: pd.Series([inner for outer in x for inner in outer])
cate_0 = set(list_flattening(training['cate_0']))
cate_1 = set(list_flattening(training['cate_1']))
cate_2 = set(list_flattening(training['cate_2']))
products = set(list_flattening(training['product']))

training.to_csv('data/processed/training.csv', index = False)
joblib.dump(cate_0, 'data/processed/cate_0.jl')
joblib.dump(cate_1, 'data/processed/cate_1.jl')
joblib.dump(cate_2, 'data/processed/cate_2.jl')
joblib.dump(products, 'data/processed/products.jl')