# _*_ coding: utf-8 _*_
#import all necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from six.moves import cPickle as pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

pickle_file = 'Xgb_99.pickle'
output_file = 'sub_99.csv'
#read dataset

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
macro = pd.read_csv('macro.csv')

train['timestamp'] = pd.to_datetime(train.timestamp)
test['timestamp'] = pd.to_datetime(test.timestamp)
macro['timestamp'] = pd.to_datetime(macro.timestamp)
print('Print the shape of train, test, and macro dataset')
print(train.shape, test.shape, macro.shape)

# fill na values with 0 in Macro dataset
macro = macro.fillna(0)
train = train.fillna(0)
test = test.fillna(0)
#macro = macro.drop(['child_on_acc_pre_school',
#                    'modern_education_share',
#                    'old_education_build_share'],axis = 1)
def rep_comma(a):
    return a.replace(',','')

macro.loc[macro.child_on_acc_pre_school == '#!','child_on_acc_pre_school'] = '0'
macro.child_on_acc_pre_school.astype(str).apply(rep_comma).astype(int)
macro.modern_education_share.astype(str).apply(rep_comma).astype(int)
macro.old_education_build_share.astype(str).apply(rep_comma).astype(int)
# encode the ctegorical variables
le = LabelEncoder()

FeatureNames = train.columns[1:-1]

Cat_features = []
for c in FeatureNames:
    if train[c].dtypes.name == 'object':
        Cat_features.append(c)
print(Cat_features)
le2 = LabelEncoder()
for cd in macro.columns:
    if macro[cd].dtypes.name =='object':
        macro.cd = le2.fit_transform(macro[cd].astype(str)).astype(int)

#impute missing values using Imputer

# impute missing values
#def impute_missing(t1,t2):
#
#    imp = Imputer(missing_values = np.nan,strategy = 'mean', axis = 1)
#    imp_cat = Imputer(missing_values = np.nan, strategy = 'most_frequent', axis = 1)
#    for col in FeatureNames:
#        if col in Cat_features:
#            try:
#                imp_cat.fit(t1[col])
#                t1[col] = imp_cat.transform(t1[col])
#                t2[col] = imp_cat.transform(t2[col])
#            except:
#                pass
#        else:
#            try:
#                imp.fit(t1[col])
#                t1[col] = imp.transform(t1[col])
#                t2[col] = imp.transform(t2[col])
#            except:
#                pass
#    return t1, t2
#train, test = impute_missing(train,test)


#change the categorical variables into labels
for col in Cat_features:
    le.fit(np.append(train[col].astype(str), test[col].astype(str)))
    train[col] = le.transform(train[col].astype(str)).astype(int)
    test[col] = le.transform(test[col].astype(str)).astype(int)


#Create some features in train and test datasets


def make_all_features(df):
    df['dayofweek'] = df.timestamp.dt.dayofweek
    df['weekofyear'] = df.timestamp.dt.weekofyear
    df['year'] = df.timestamp.dt.year
    df['month'] = df.timestamp.dt.month
    df['inv_floor'] = 1./df.floor
    return df

train = make_all_features(train)
test = make_all_features(test)

def get_macro_features(df):

    return pd.merge(df, macro, on='timestamp', how='left')

train_macro =get_macro_features(train)
test_macro = get_macro_features(test)


#define the train data test data and train label
X = train_macro.drop(['id','timestamp','price_doc'],axis = 1)
y = np.log(train_macro.price_doc)

X_new = test_macro.drop(['id','timestamp'],axis=1)

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
