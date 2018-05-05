__author__ = 'ktutuncu'

import numpy as np
import scipy as sp
import pandas as pd
import os
import sklearn as sk
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import sklearn.linear_model as lm
from sklearn.feature_extraction import DictVectorizer


def read_cleanup_etc(path, data_filename, nrows, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=[], target=None):
    df = pd.read_csv(path + data_filename, nrows = nrows, skiprows = skiprows)
    fields = pd.read_csv(path + fields_filename, header=None)
    df.columns = list(fields.iloc[0,:])

    targets = ['click', 'zero', 'twentyfive', 'fifty', 'seventyfive', 'hundred', 'audience_segments']
    y_target = None
    if target is not None:
        if target not in targets:
            print "Target variable not valid."
            exit(0)
        else:
            y_target = df[target]

    for col in cols_to_drop + targets:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    df.dropna (axis=1, how='any', inplace=True)
    #len(df.columns)
    return df, np.array(y_target)


path = '/home/ktutuncu/Desktop/'
data_filename = 'b9a3fcd9-68fa-4c7a-890c-0d026bdd2024-000000'
fields_filename = 'FieldNames.txt'

nrows = 1000
target = 'hundred'

x_cat,y = read_cleanup_etc(path, data_filename, nrows, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=['event_time_stamp'], target=target)

#testing df
# sum(df['click'])
# sum(df['hundred'])

# y_click = np.array(df.loc['click'])
# y_100  = np.array(df.loc['hundred'])
# x_train  = np.array(df.drop(['click', 'zero', 'twentyfive', 'fifty', 'seventyfive', 'hundred'], axis=1))

vectorizer = DictVectorizer()
x = vectorizer.fit_transform(x_cat.T.to_dict().values()).toarray()
y

x = abs(x)
x_new = SelectKBest(chi2, k=100).fit_transform(x, y)

x_new.shape
