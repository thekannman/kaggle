import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from random import sample


class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

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
        self.__trainlabels = np.ones((self.__traindata.shape[0],))
        data = np.empty((0, driver.num_features), float)
        setkeys = datadict.keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        else:
            setkeys = sample(setkeys, len(setkeys) - 1)
        for key in setkeys:
            if key != driver.identifier:
                data = np.append(data, np.asarray(datadict[key]), axis=0)
        self.__traindata = np.append(self.__traindata, data, axis=0)
        self.__trainlabels = np.append(self.__trainlabels, np.zeros((data.shape[0],)), axis=0)
        self.__y = np.ones((self.__testdata.shape[0],))

    def classify(self):
        """Perform classification"""
        clf = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        clf.fit(self.__traindata, self.__trainlabels)
        self.__y = clf.predict(self.__testdata)

    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        for i in xrange(len(self.__indexlist) - 1):
            returnstring += "%d_%d,%.3f\n" % (self.driver.identifier, self.__indexlist[i], self.__y[i])
        returnstring += "%d_%d,%.3f" % (self.driver.identifier, self.__indexlist[len(self.__indexlist)-1], self.__y[len(self.__indexlist)-1])
        return returnstring
