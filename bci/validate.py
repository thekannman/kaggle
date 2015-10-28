from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier, GradientBoostingRegressor as GBRegressor, RandomForestClassifier as RFClassifier
from sklearn.linear_model import Lasso,ElasticNet,Ridge,BayesianRidge,LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

class RFClassifierWithCoef(RFClassifier):
    def fit(self, *args, **kwargs):
        super(RFClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

def roc_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print auc


def leavePLabelOut(X, y_true,labels):
    y_pred = y_true * 0.0
    lpl = cross_validation.LeavePLabelOut(labels, p=1)
    i = 0
    for train_index, test_index in lpl:
        i += 1
        print "CrossVal "+str(i)+" of "+str(len(lpl))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
        clf = RFClassifier(n_estimators=200)
        #clf = GBClassifier(n_estimators=5000,learning_rate=0.05, max_features=0.25)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict_proba(X_test)
    roc_auc(y_true, y_pred)

def KFold(X, y_true):
    y_pred = y_true * 0.0
    y_pred1 = y_true * 0.0
    y_pred2 = y_true * 0.0
    y_pred3 = y_true * 0.0
    y_pred4 = y_true * 0.0
    y_pred5 = y_true * 0.0

    kf = cross_validation.StratifiedKFold(y_true, n_folds = 2)
    i = 0
    for train_index, test_index in kf:
        i += 1
        print "CrossVal "+str(i)+" of "+str(len(kf))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]
        pca = PCA(n_components = 100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
       
        #clf1 = RFClassifierWithCoef(n_estimators=500,n_jobs=2)
        #selector = RFE(clf1, step=100)
        #clf2 = GBClassifier(n_estimators=500,learning_rate=0.05, max_features=0.25)
        #clf1.fit(X_train, y_train)
        #clf2.fit(X_train, y_train)
        #y_pred1[test_index] = clf1.predict_proba(X_test)  
        #y_pred2[test_index] = clf3.predict_proba(X_test)
        #y_pred = (y_pred1 + y_pred2)/2.0 
        #lasso = Lasso(alpha=0.1, max_iter=1000000) #Maybe best so far
        #clf1 = RFE(lasso, step=1) 
        #selector.fit(X_train,y_train)
        #X_test = selector.transform(X_test)
        parameters = {'alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5.0,10.0],'l1_ratio':[0.0,0.25,0.5,0.75,1.0]}
        elast = ElasticNet(max_iter=10000000) 
        clf2 = GridSearchCV(elast, parameters,scoring='roc_auc')
        #clf3 = BayesianRidge() 
        #parameters = {'alpha':[0.001,0.01,0.1,1.0,10.0]}#,'l1_ratio':[0.0,0.25,0.5,0.75,1.0]}
        #clf3 = Ridge(alpha=1.0)
        #clf2 = GridSearchCV(linear, parameters,scoring='roc_auc')
        #lasso = Lasso()
        #clf2 = GridSearchCV(lasso, parameters,scoring='roc_auc')
        #linear = LinearRegression()
        #clf2 = RFE(linear, step=1, n_features_to_select=100)
        #clf4 = KNNRegressor(n_neighbors = 10, weights='uniform')
        #clf2 = SVR(kernel = 'linear', cache_size=2000, verbose=True)
        print 'Lasso'
        clf2.fit(X_train, y_train)
        print 'ElasticNet'
        #clf2.fit(X_train, y_train)
        print 'Ridge'
        #clf3.fit(X_train, y_train)
        print clf2.best_params_
        print 'KNN'
        #clf4.fit(X_train, y_train)
        print 'SVR'
        #clf5.fit(X_train, y_train)
        #y_pred1[test_index] = clf1.predict(X_test)
        y_pred2[test_index] = clf2.predict(X_test)
        #y_pred3[test_index] = clf3.predict(X_test)
        #y_pred4[test_index] = clf4.predict(X_test)
        #y_pred5[test_index] = clf5.predict(X_test)
        #y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5)/5.0 
    #print y_pred1
    #y_pred1 = y_pred1[:,1]
    #roc_auc(y_true, y_pred1)
    roc_auc(y_true, y_pred2)
    #roc_auc(y_true, y_pred3)
    #roc_auc(y_true, y_pred4)
    #roc_auc(y_true, y_pred5)
    #roc_auc(y_true, y_pred)
    exit()

