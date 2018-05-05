                          __author__ = 'rgerami'

import numpy as np
import pandas as pd
import pickle
import sklearn as sk
import sklearn.linear_model as lm
from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



path = '/home/ktutuncu/Desktop/'
data_filename = 'b9a3fcd9-68fa-4c7a-890c-0d026bdd2024-000000'
fields_filename = 'FieldNames.txt'

nrows = 100
skiprows=0
cols_to_drop = []
target = 'hundred'

x_cat, y = read_cleanup_etc(path, data_filename, nrows, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=[], target=target)

vectorizer = DictVectorizer()
x = vectorizer.fit_transform(x_cat.T.to_dict().values()).toarray()

# tmp = lm.LogisticRegression()
# tmp.fit(x,y)
tmp = SelectKBest(chi2, k=20)
x_new = tmp.fit(x, y)





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


def perf_feature_vectorizer (path, data_filename, nrows, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=[]):
    df = read_cleanup_etc (path, data_filename, nrows, skiprows, 'FieldNames.txt', cols_to_drop)[0]
    vectorizer = DictVectorizer()
    vectorizer.fit(df.T.to_dict().values())
    pickle.dump(vectorizer, open(path + 'vectorizer.pickle', 'w'))
    return vectorizer

def perf_train(target, path, data_filename, nrows, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=[]):
    df, y_train = read_cleanup_etc (path, data_filename, nrows, skiprows, 'FieldNames.txt', cols_to_drop, target)
    vectorizer = pickle.load(open(path + 'vectorizer.pickle', 'r'))
    x_train = vectorizer.transform(df.T.to_dict().values())
    model = lm.SGDClassifier(loss='log', penalty='l2')
    model.fit(x_train, y_train)
    pickle.dump(model, open(path + 'logreg_' + target +'.pickle', 'w'))
    return model

def perf_forecast (target, path, data_filename, nrows = 1, skiprows=0, fields_filename = 'FieldNames.txt', cols_to_drop=[]):
    df, y_train = read_cleanup_etc (path, data_filename, nrows, skiprows, 'FieldNames.txt', cols_to_drop, target)
    vectorizer = pickle.load (open(path + 'vectorizer.pickle', 'r'))
    model =      pickle.load (open(path + 'logreg_' + target +'.pickle', 'r'))
    x = vectorizer.transform(df.T.to_dict().values())
    y_pred = model.predict(x)
    print (y_pred)
    return y_pred

x, y
model = lm.LogisticRegression(penalty='l1')
model.fit(x, y)
x_new = model.transform(x)


