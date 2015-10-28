# $Id: ClassifyDriver.py,v 1.8 2015/03/16 14:54:26 zak Exp zak $

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier, GradientBoostingRegressor as GBRegressor, RandomForestClassifier as RFClassifier, RandomForestRegressor as RFRegressor, ExtraTreesClassifier as ETClassifier, ExtraTreesRegressor as ETRegressor
from sklearn.linear_model import Lasso,ElasticNet,Ridge,LinearRegression,LogisticRegression,SGDClassifier
from sklearn.svm import SVC,SVR,OneClassSVM
from random import sample
from sklearn.feature_selection import RFE,RFECV
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.grid_search import GridSearchCV


class RFClassifierWithCoef(RFClassifier):
    def fit(self, *args, **kwargs):
        super(RFClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class ETClassifierWithCoef(ETClassifier):
    def fit(self, *args, **kwargs):
        super(ETClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class ClassifyDriver(object):
    """Not useful directly, but allows below classes to inherit otherwise redundant portions."""

    def __init__(self, driver, datadict):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        self.driver = driver
        self.numfeatures = self.driver.num_features
        featurelist = []
        self.__indexlist = []
        for trace in self.driver.traces:
            self.__indexlist.append(trace.identifier)
            featurelist.append(trace.features)
        # Initialize train and test np arrays
        self.__traindata = np.asarray(featurelist)
        self.__testdata = np.asarray(featurelist)
        #ocSVM = OneClassSVM(kernel='poly', nu=0.1)
        #ocSVM.fit(self.__traindata)
        #self.__trainlabels = ocSVM.predict(self.__traindata)
        #pca = PCA(n_components = 50)
        #new_train_data = pca.fit_transform(self._ClassifyDriver__traindata)
        #elliptic_envelope = EllipticEnvelope(contamination=0.025)
        #elliptic_envelope.fit(new_train_data)
        #self.__trainlabels = elliptic_envelope.predict(new_train_data)
        #for i in range(len(self.__trainlabels)):
        #    if self.__trainlabels[i] == -1:
        #        self.__trainlabels[i] = 0
        #print self.__trainlabels
        #print sum(self.__trainlabels)
        self.__trainlabels = np.ones((self.__traindata.shape[0],))
        data = np.empty((0, driver.num_features), float)
        setkeys = datadict.keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        else:
            setkeys = sample(setkeys, len(setkeys) - 1)
        for key in setkeys:
            if key != driver.identifier:
                weighted_datadict = sample(datadict[key],len(datadict[key])/len(setkeys))
                data = np.append(data, np.asarray(weighted_datadict), axis=0)
        self.__traindata = np.append(self.__traindata, data, axis=0)
        self.__trainlabels = np.append(self.__trainlabels, np.zeros((data.shape[0],)), axis=0)
        self.__y = np.ones((self.__testdata.shape[0],))

    def classify(self):
        """Placeholder for classifiers in inherited classes"""
        print "Must pick classifier type."
        exit()
    
    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        for i in xrange(len(self.__indexlist) - 1):
            returnstring += "%d_%d,%.3f\n" % (self.driver.identifier, self.__indexlist[i], self.__y[i])
        returnstring += "%d_%d,%.3f" % (self.driver.identifier, self.__indexlist[len(self.__indexlist)-1], self.__y[len(self.__indexlist)-1])
        return returnstring

class RegressionDriver(ClassifyDriver):
    """Class for Regression-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = LinearRegression()
        pca = PCA(n_components = 50)
        self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)        
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class LogRegDriver(ClassifyDriver):
    """Class for Regression-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = LogisticRegression()
        pca = PCA(n_components = 50)
        self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class SGDDriver(ClassifyDriver):
    """Class for Regression-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = SGDClassifier(loss='log', penalty='l1')
        pca = PCA(n_components = 10)
        self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict_proba(self._ClassifyDriver__testdata)[:,0]
        #rfe = RFE(estimator=clf, step=10)
        #rfe.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        #plt.figure()
        #plt.xlabel("Number of features selected")
        #plt.ylabel("Cross validation score (nb of correct classifications)")
        #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        #plt.show()
        #self._ClassifyDriver__y = rfe.predict_proba(self._ClassifyDriver__testdata)[:,0]


class RandomForestDriver(ClassifyDriver):
    """Class for RandomForest-based analysis of Driver traces"""
    """Removed max_depth and added compute_importances at 11:39, 2015/2/13"""
    def classify(self):
        """Perform classification"""
        clf = RFRegressor(n_estimators=5000, min_samples_split=10, min_samples_leaf=2, max_depth=6)
        #pca = PCA(n_components = 400)
        #self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        #self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)        
        #print self._ClassifyDriver__traindata.shape
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)
        #self._ClassifyDriver__y = clf.predict_proba(self._ClassifyDriver__testdata)[:,0]
        #rfecv = RFECV(estimator=clf, step=100, cv=StratifiedKFold(self._ClassifyDriver__trainlabels, 4), scoring='roc_auc')
        #rfecv.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        #plt.figure()
        #plt.xlabel("Number of features selected")
        #plt.ylabel("Cross validation score (nb of correct classifications)")
        #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        #plt.show()
        #self._ClassifyDriver__y = rfecv.predict_proba(self._ClassifyDriver__testdata)[:,0]

class ExtraTreesDriver(ClassifyDriver):
    """Class for ExtraTrees-based analysis of Driver traces"""
    def classify(self):
        """Perform classification"""
        clf = ETRegressor(n_estimators=500, min_samples_split=5, min_samples_leaf=2)
        #pca = PCA(n_components = 400)
        #self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        #self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)
        #print self._ClassifyDriver__traindata.shape
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)
        #self._ClassifyDriver__y = clf.predict_proba(self._ClassifyDriver__testdata)[:,0]
        #rfe = RFE(estimator=clf, step=10)
        #rfecv = RFECV(estimator=clf, step=50, cv=StratifiedKFold(self._ClassifyDriver__trainlabels, 4), scoring='roc_auc')
        #rfe.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        #plt.figure()
        #plt.xlabel("Number of features selected")
        #plt.ylabel("Cross validation score (nb of correct classifications)")
        #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        #plt.show()
        #self._ClassifyDriver__y = rfe.predict_proba(self._ClassifyDriver__testdata)[:,0]


class LassoDriver(ClassifyDriver):
    """Class for Lasso-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = Lasso(max_iter=10000000)
        #parameters = {'alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5.0,10.0]}
        #clf = GridSearchCV(lasso, parameters,scoring='roc_auc')

        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class ElasticNetDriver(ClassifyDriver):
    """Class for ElasticNet-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = ElasticNet()
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class RidgeDriver(ClassifyDriver):
    """Class for Ridge-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = Ridge()
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class LinearSVCDriver(ClassifyDriver):
    """Class for Ridge-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = SVR(kernel = 'poly', degree=3)
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class RBFSVRDriver(ClassifyDriver):
    """Class for Ridge-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = SVR(kernel = 'rbf')
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict(self._ClassifyDriver__testdata)

class GBDriver(ClassifyDriver):
    """Class for GradientBoostingClassifier-based analysis of Driver traces"""

    def classify(self):
        """Perform classification"""
        clf = GBClassifier(n_estimators=10000,learning_rate=0.1, max_features='auto', min_samples_split=10, min_samples_leaf=2, max_depth=6)
        #pca = PCA(n_components = 50)
        #self._ClassifyDriver__traindata = pca.fit_transform(self._ClassifyDriver__traindata)
        #self._ClassifyDriver__testdata = pca.transform(self._ClassifyDriver__testdata)   
        clf.fit(self._ClassifyDriver__traindata, self._ClassifyDriver__trainlabels)
        self._ClassifyDriver__y = clf.predict_proba(self._ClassifyDriver__testdata)[:,0]

