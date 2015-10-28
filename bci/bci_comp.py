## author: Zachary Kann
## kaggle bci challenge
## Special thanks to phalaris for his gbm benchmark code 

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from parser import parser
import validate


class RFCCoef(RFClassifier):
    """Modifies the Random Forest Classifier of sklearn to allow use of 
    recursive feature elimination with cross-validation
    """
    def fit(self, *args, **kwargs):
        super(RFCCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


options = parser()
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')
# Choose portion of signal to use after feedback event
begin = 1
end = 260
submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
features = ['Cz', 'FC1','Fz','POz','P7', 'P8']


def read_dataframe(purpose, indices):
    """Imports metadata and features from the csv file.
    Returns dataframe consisting of said data.
    """

    frame = pd.DataFrame(columns=['subject','session',
                                  'feedback_num','start_pos'] + 
                         [feature+'_' + s for feature in features 
                             for s in map(str,range(begin,end+1))],
                         index=range(indices))    
    counter = 0
    print 'loading '+purpose+' data'
    data = {}
    filenames = os.listdir('data/'+purpose)
    print filenames
    numfiles = len(filenames)
    filecounter = 0
    for filename in filenames:
        filecounter += 1
        split = filename.split('_')
        subjectID = re.sub("[^[0-9]", "", split[1])
        sessionID = re.sub("[^[0-9]", "", split[2])
        temp = pd.read_csv("data/"+purpose+"/" + filename)
        # Select rows with positive feedback result
        fb = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']
        counter2 = 0
        for index in fb.index:
            for feature in features:
                start_value = temp.loc[int(index)+1,feature]
                temp2 = temp.loc[int(index)+begin:int(index)+end,feature]
                temp2.index = [feature+'_' + 
                               s for s in map(str,range(begin,end+1))]
                for key in temp2.keys():
                    temp2[key] -= start_value
                frame.loc[counter,[feature+'_' + 
                    s for s in map(str,range(begin,end+1))]] = temp2
            frame.loc[counter,'session'] = sessionID
            frame.loc[counter, 'subject'] = subjectID
            frame.loc[counter, 'feedback_num'] = counter2
            frame.loc[counter, 'start_pos'] = index
            counter += 1
            counter2 += 1
        print purpose+' %d of %d' % (filecounter, numfiles)
    return frame


if options.continued:
    print "Using old train/test results"
    train = pd.read_csv(options.train, index_col=0)
    test = pd.read_csv(options.test, index_col=0)
else:
    # Read training data
    train = read_dataframe('train', 5440)
    train_csv = "train_{0}.csv".format(submission_id)
    train.to_csv(train_csv,ignore_index=True)
    # Read testing data
    test = read_dataframe('test', 3400)
    test_csv = "test_{0}.csv".format(submission_id)
    test.to_csv(test_csv, ignore_index=True)

if options.append_sum:
    print "Appending sum to train"
    feature_sum = [0.0]*len(train.index)
    feature_diff = [0.0]*len(train.index)
    feature_old = [0.0]*len(train.index)
    for feature in features:
        colname = feature+'_1'
        for index in train.index:
            feature_val = train[colname][index]
            feature_sum[index] = feature_val
            feature_old[index] = feature_val
        for i in map(str,range(begin+1,end+1)):
            colname = feature+'_'+i
            for index in train.index:
                feature_val = train[colname][index]
                feature_sum[index] += feature_val
                feature_diff[index] = feature_val - feature_old[index]
                feature_old[index] = feature_val
            sumcol = feature+'_sum_'+i
            diffcol = feature+'_diff_'+i
            train[sumcol] = feature_sum
            train[diffcol] = feature_diff
    train_csv = "train_sumdiff_{0}.csv".format(submission_id)
    train.to_csv(train_csv,ignore_index=True)

    print "Appending sum to test"
    feature_sum = [0.0]*len(test.index)
    feature_diff = [0.0]*len(test.index)
    feature_old = [0.0]*len(test.index)
    for feature in features:
        colname = feature+'_1'
        for index in test.index:
            feature_val = test[colname][index]
            feature_sum[index] = feature_val
            feature_old[index] = feature_val
        for i in map(str,range(begin+1,end+1)):
            colname = feature+'_'+i
            for index in test.index:
                feature_val = test[colname][index]
                feature_sum[index] += feature_val
                feature_diff[index] = feature_val - feature_old[index]
                feature_old[index] = feature_val
            sumcol = feature+'_sum_'+i
            diffcol = feature+'_diff_'+i
            test[sumcol] = feature_sum
            test[diffcol] = feature_diff
    test_csv = "test_sumdiff_{0}.csv".format(submission_id)
    test.to_csv(test_csv,ignore_index=True)

if options.dropout:
    for feature in features:
        for i in map(str,range(begin+1,end+1)):
            print "working on "+feature+" number "+i
            if int(i) % 5 == True: continue
            drop_cols = [feature+'_'+i, feature+'_sum_'+i]
            train = train.drop(drop_cols, axis=1)
            test = test.drop(drop_cols, axis=1)
    train_csv = "train_dropout_{0}.csv".format(submission_id)
    train.to_csv(train_csv,ignore_index=True)
    test_csv = "test_dropout_{0}.csv".format(submission_id)
    test.to_csv(test_csv,ignore_index=True)

#Validate
if (options.validate):
    y_true = train_labels.values[:,1].ravel().astype(int)
    validate.KFold(train.values[:,4:], y_true)
    #validate.KFold(train.drop('subject', axis=1).values, y_true)

X_train = train.values[:,4:]
X_test = test.values[:,4:]
pca = PCA(n_components=70)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print 'training'
output = "py_{0}.csv".format(submission_id)
clf = LinearRegression()
y_true = train_labels.values[:,1].ravel().astype(int)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(y_true, 4),
              scoring='roc_auc')
rfecv.fit(X_train, y_true)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

preds = rfecv.predict(X_test)
submission['Prediction'] = preds
submission.to_csv(output,index=False)
print 'Done'
