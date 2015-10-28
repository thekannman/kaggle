# coding: utf-8

"""
Beating the Benchmark 
BCI Challenge @ Kaggle

__author__ : rcarson (https://twitter.com/daxiongshu)
__original_author__ :Abhishek (abhishek4 AT gmail)
"""

import numpy as np
import pandas as pd
from sklearn import ensemble
from xgb_classifier import xgb_classifier
labels = pd.read_csv('../../data/TrainLabels.csv')
submission = pd.read_csv('../../data/SampleSubmission.csv')

training_files = []
for filename in labels.IdFeedBack.values:
    training_files.append(filename[:-6])  

testing_files = []
for filename in submission.IdFeedBack.values:
    testing_files.append(filename[:-6])  

for i, filename in enumerate(np.unique(training_files)):
    print i, filename
    path = '../../data/train/Data_' + str(filename) + '.csv'
    df = pd.read_csv(path)
    df = df[df.FeedBackEvent != 0]
    df = df.drop('FeedBackEvent', axis = 1)
    if i == 0:
        train = np.array(df)
    else:
        train = np.vstack((train, np.array(df)))

for i, filename in enumerate(np.unique(testing_files)):
    print i, filename
    path = '../../data/test/Data_' + str(filename) + '.csv'
    df = pd.read_csv(path)
    df = df[df.FeedBackEvent != 0]
    df = df.drop('FeedBackEvent', axis = 1)
    if i == 0:
        test = np.array(df)
    else:
        test = np.vstack((test, np.array(df)))
"""
import pickle
pickle.dump(train,open("train.p","wb"))
pickle.dump(test,open("test.p","wb"))
pickle.dump(labels.Prediction.values,open("label.p","wb"))

import pickle
train=pickle.load(open("train.p","rb"))
test=pickle.load(open("test.p","rb"))
label=pickle.load(open("label.p","rb"))
"""
clf = ensemble.RandomForestClassifier(n_jobs = -1, 
				     n_estimators=10,
                                     min_samples_leaf=10, 
			             random_state=42)
xgb_clf=xgb_classifier(eta=0.1,min_child_weight=1,depth=10,num_round=40,threads=8,boost_from_exist_prediction=True,exist_num_round=10)
clf.fit(train, labels.Prediction.values)
base_train_prediction=clf.predict_proba(train).T[1]
base_test_prediction=clf.predict_proba(test).T[1]
preds = xgb_clf.train_predict(train,labels.Prediction.values,test,base_train_prediction,base_test_prediction)

submission['Prediction'] = preds
submission.to_csv('xgb_boost_from_rf.csv', index = False)
