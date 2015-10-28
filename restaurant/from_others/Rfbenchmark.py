#############################################################
# developed by rcarson
# aixueer4ever@gmail.com
# based on Dmitry Dryomov's tradeshift benchmark
############################################################

from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
data_dir='../../data/'
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
vec = DictVectorizer()

names_categorical = []
for name in train.columns[1:-1] :    
    if name.startswith('P') == False:

        names_categorical.append(name)
        print name, len(np.unique(train[name]))
X_train_cat=vec.fit_transform(train[names_categorical].T.to_dict().values()).todense()
X_test_cat = vec.transform(test[names_categorical].T.to_dict().values()).todense()

numerical_label=['P'+str(i) for i in range(1,38)]
X_train_num=train[numerical_label]
X_test_num=test[numerical_label]

X=np.hstack((X_train_cat, X_train_num))
Xt=np.hstack((X_test_cat,X_test_num))
y=train['revenue']

print X.shape, Xt.shape, y.shape

clf=RandomForestRegressor(n_estimators=1000)
clf.fit(X,y)
yp=clf.predict(Xt)

sub=pd.read_csv(data_dir + 'sampleSubmission.csv')
sub['Prediction']=yp
sub.to_csv('sub1.csv',index=False)
