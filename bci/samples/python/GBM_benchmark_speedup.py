## author: phalaris
## edited by: Brandon Veber
##  Uses numpy array for file retrieval
## kaggle bci challenge gbm benchmark

from __future__ import division
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
import time

tstart=time.time()
train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
test_subs = ['01','03','04','05','08','09','10','15','19','25']
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')
### Variable Parameters
nSecs=1.3
electrodes = ['Cz']
###################33##
nPts = int(nSecs*200)
columns = ['subject','session','feedback_num','start_pos']
for electrode in electrodes:
    columns += [electrode+'_' + s for s in map(str,range(nPts+1))]
train = np.empty((5440,4+(nPts+1)*len(electrodes)),dtype=np.float32)
counter = 0
print('loading train data')
data = {}
t0=time.time()
for i in train_subs:
    print('subject ', i)
    t1 = time.time()
    for j in range(1,6):
        t2 = time.time()
        temp = pd.read_csv('train/Data_S' + i + '_Sess0'  + str(j) + '.csv')
        fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']        
        counter2 = 0
        for k in fb.index:
            row = np.array([np.float32(i),np.float32(j),np.float32(counter2),np.float32(k)])
            for electrode in electrodes:
                temp2 = list(temp.loc[int(k):int(k)+nPts,electrode])
                row = np.append(row,temp2-np.mean(temp2))
            train[counter] = row
            counter +=1
            counter2 +=1
pd.DataFrame(train,columns=columns).to_csv('train_'+'_'.join(electrodes)+'.csv',ignore_index=True)

test = np.empty((3400,4+(nPts+1)*len(electrodes)),dtype=np.float32)
print('loading test data')
counter = 0
data = {}
t0=time.time()
for i in test_subs:
    print('subject ', i)
    t1 = time.time()
    for j in range(1,6):
        temp = pd.read_csv('test/Data_S' + i + '_Sess0'  + str(j) + '.csv')
        fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']        
        counter2 = 0
        for k in fb.index:
            row = np.array([np.float32(i),np.float32(j),np.float32(counter2),np.float32(k)])
            for electrode in electrodes:
                temp2 = list(temp.loc[int(k):int(k)+nPts,electrode])
                row = np.append(row,temp2-np.mean(temp2))
            test[counter] = row
            counter +=1  
            counter2 +=1
pd.DataFrame(test,columns=columns,index=range(3400)).to_csv('test_'+'_'.join(electrodes)+'.csv',ignore_index=True)

print('training GBM')

gbm = ens.GradientBoostingClassifier(n_estimators=500,learning_rate=0.05, max_features=0.25)
gbm.fit(train, train_labels.values[:,1].ravel())
preds = gbm.predict_proba(test)
preds = preds[:,1]
submission['Prediction'] = preds
submission.to_csv('gbm_benchmark.csv',index=False)
print('Done')
print('Total Runtime is ',time.time()-tstart)