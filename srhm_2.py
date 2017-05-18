# https://github.com/dmlc/xgboost/issues/463#issuecomment-147365960
#http://stackoverflow.com/questions/40556057/oserror-when-importing-xgboost

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from six.moves import cPickle as pickle
import os


train_df = pd.read_csv("train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("test.csv", parse_dates=['timestamp'])
macro_df = pd.read_csv("macro.csv", parse_dates=['timestamp'])
train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')
test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')
print(train_df.shape, test_df.shape)

# truncate the extreme values in price_doc #
ulimit = np.percentile(train_df.price_doc.values, 99)
llimit = np.percentile(train_df.price_doc.values, 1)
train_df.loc[train_df['price_doc']>ulimit,'price_doc'] = ulimit
train_df.loc[train_df['price_doc']<llimit,'price_doc'] = llimit

#remove the special characters first
try:
    train_df.loc[train_df['child_on_acc_pre_school']== '#!','child_on_acc_pre_school'] = '0'
    test_df.loc[test_df['child_on_acc_pre_school']== '#!','child_on_acc_pre_school'] = '0'
except:
    pass

def rep_commas(a):
    return a.replace(',','')

train_df.child_on_acc_pre_school= train_df.child_on_acc_pre_school.fillna(0).astype(str).apply(rep_commas).astype(int)
train_df.modern_education_share = train_df.modern_education_share.fillna(0).astype(str).apply(rep_commas).astype(int)
train_df.old_education_build_share  =train_df.old_education_build_share.fillna(0).astype(str).apply(rep_commas).astype(int)
test_df.child_on_acc_pre_school= test_df.child_on_acc_pre_school.fillna(0).astype(str).apply(rep_commas).astype(int)
test_df.modern_education_share = test_df.modern_education_share.fillna(0).astype(str).apply(rep_commas).astype(int)
test_df.old_education_build_share  =test_df.old_education_build_share.fillna(0).astype(str).apply(rep_commas).astype(int)

for f in train_df.columns:
    if train_df[f].dtype=='object':
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))
        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))
        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))

def add_count(df, group_col):
    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()
    grouped_df.columns = [group_col, "count_"+group_col]
    df = pd.merge(df, grouped_df, on=group_col, how="left")
    return df


def make_features(df):
    df["null_count"] = df.isnull().sum(axis=1)
    df.fillna(-99, inplace=True)
    # year and month #
    df["yearmonth"] = df["timestamp"].dt.year*100 + df["timestamp"].dt.month
    # year and week #
    df["yearweek"] = df["timestamp"].dt.year*100 + df["timestamp"].dt.weekofyear
    # year #
    df["year"] = df["timestamp"].dt.year
    # month of year #
    df["month_of_year"] = df["timestamp"].dt.month
    # week of year #
    df["week_of_year"] = df["timestamp"].dt.weekofyear
    # day of week #
    df["day_of_week"] = df["timestamp"].dt.weekday
    # ratio of living area to full area #
    df["ratio_life_sq_full_sq"] = df["life_sq"] / np.maximum(df["full_sq"].astype("float"),1)

    df.loc[df["ratio_life_sq_full_sq"]<0,"ratio_life_sq_full_sq"] = 0
    df.loc[df["ratio_life_sq_full_sq"]>1,"ratio_life_sq_full_sq"] = 1
    # ratio of kitchen area to living area #
    df["ratio_kitch_sq_life_sq"] = df["kitch_sq"] / np.maximum(df["life_sq"].astype("float"),1)

    df.loc[df["ratio_kitch_sq_life_sq"]<0,"ratio_kitch_sq_life_sq"] = 0
    df.loc[df["ratio_kitch_sq_life_sq"]>1,"ratio_kitch_sq_life_sq"] = 1
    # ratio of kitchen area to full area #
    df["ratio_kitch_sq_full_sq"] = df["kitch_sq"] / np.maximum(df["full_sq"].astype("float"),1)

    df.loc[df["ratio_kitch_sq_full_sq"]<0,"ratio_kitch_sq_full_sq"] = 0
    df.loc[df["ratio_kitch_sq_full_sq"]>1,"ratio_kitch_sq_full_sq"] = 1
    # floor of the house to the total number of floors in the house #
    df["ratio_floor_max_floor"] = df["floor"] / df["max_floor"].astype("float")

    # num of floor from top #
    df["floor_from_top"] = df["max_floor"] - df["floor"]
    df["extra_sq"] = df["full_sq"] - df["life_sq"]
    df["age_of_building"] = df["build_year"] -df["year"]
    df["ratio_preschool"] = df["children_preschool"] / df["preschool_quota"].astype("float")
    df["ratio_school"] = df["children_school"] / df["school_quota"].astype("float")
    df["ratio_preschool"] = df["children_preschool"] / df["preschool_quota"].astype("float")
    df["ratio_school"] = df["children_school"] / df["school_quota"].astype("float")
    df = add_count(df, "yearmonth")
    df = add_count(df, "yearweek")

    return df

train_df = make_features(train_df)
test_df = make_features(test_df)

train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
test_X = test_df.drop(["id", "timestamp"] , axis=1)

train_y = np.log1p(train_df.price_doc.values)

#cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
#ind_params = {'learning_rate': 0.1,
#              'n_estimators': 1000,
#              'seed':0,
#              'subsample': 0.8,
#              'colsample_bytree': 0.8,
#            'objective': 'reg:linear'}
cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8,
                           'objective': 'reg:linear', 'max_depth': 3, 'min_child_weight': 1}
optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params), cv_params,
                             scoring = 'neg_mean_squared_error',
                             cv = 5, n_jobs = -1)
optimized_GBM.fit(train_X, train_y)

print(optimized_GBM.grid_scores_)

# Best score is achieved with max_depth = 3 and min_child_weight = 1

