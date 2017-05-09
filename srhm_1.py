# _*_ coding: utf-8 _*_
#import all necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from six.moves import cPickle as pickle
import os

pickle_file = 'Xgb_3.pickle'
output_file = 'sub_3.csv'
#read dataset

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
macro = pd.read_csv('macro.csv')

print('Print the shape of train, test, and macro dataset')
print(train.shape, test.shape, macro.shape)


#fill NaN values with 0

train = train.fillna(0)
test = test.fillna(0)

# encode the ctegorical variables


le = LabelEncoder()

FeatureNames = train.columns[1:-1]

Cat_features = []
for c in FeatureNames:
    if train[c].dtypes.name == 'object':
        Cat_features.append(c)
print(Cat_features)

for col in Cat_features:
    le.fit(np.append(train[col].astype(str), test[col].astype(str)))
    train[col] = le.transform(train[col].astype(str)).astype(int)
    test[col] = le.transform(test[col].astype(str)).astype(int)

#define the train data test data and train label
X = train.drop(['id','timestamp','price_doc'],axis = 1)
y = np.log(train.price_doc)

X_new = test.drop(['id','timestamp'],axis=1)

xgb_model = xgb.XGBRegressor()
param_grid = {}
param_grid['max_depth'] = [4,6,8]
param_grid['n_estimators'] = [100,150,200]
param_grid['learning_rate'] = [0.01,0.05,0.07,0.1]

if not os.path.isfile(pickle_file):
    clf = GridSearchCV(xgb_model,param_grid,scoring = 'r2',verbose = 1,cv = 5,
                       n_jobs = 8)
    clf.fit(X,y)
    try:
        with open(pickle_file,'wb') as f:
            pickle.dump(clf,f,pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save output to ',pickle_filename, ':',e)


else:
    #Pickle the output
    with open(pickle_file,'rb') as f:
        save = pickle.load(f)
        clf = save['clf']
print(clf.best_score_)
print(clf.best_params_)

y_pred = clf.predict(X_new)

y_out = np.exp(y_pred)

#Create a DataFrame that only contains the IDs and predicted values
if not os.path.isfile(output_file):
    pd.DataFrame({'id':test.id, 'price_doc': y_out}).set_index('id').to_csv(output_file)
else:
    print('Output file', output_file, ' already exists into the location')
