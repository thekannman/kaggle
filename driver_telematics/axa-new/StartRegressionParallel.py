__version__ = '$Revision: 1.7 $'
# $Id: StartRegressionParallel.py,v 1.7 2015/03/12 05:44:10 zak Exp zak $

"""Main module for Kaggle AXA Competition

Uses the logistic regression idea described by Stephane Soulier: https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/11299/score-0-66-with-logistic-regression
Hence, we use the traces from every driver as positive examples and build a set of references that we use as negative examples. Note that our set is larger by one driver, in case the reference set includes the driver that we are currently using as positive.
"""

from datetime import datetime
from Driver import Driver
from ClassifyDriver import RegressionDriver,RandomForestDriver,ExtraTreesDriver,LassoDriver,ElasticNetDriver,RidgeDriver,LinearSVCDriver,RBFSVRDriver,GBDriver,LogRegDriver,SGDDriver
import os
import sys
from random import sample, seed
from joblib import Parallel, delayed
import re

REFERENCE_DATA = {}

def generatedata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global REFERENCE_DATA
    for driver in drivers:
        REFERENCE_DATA[driver.identifier] = driver.generate_data_model


def perform_analysis(folder, model_choice):
    print "Working on {0}".format(folder)
    sys.stdout.flush()
    temp = Driver(folder)
    if model_choice == 'Regression':
        cls = RegressionDriver(temp, REFERENCE_DATA)
    elif model_choice == 'LogReg':
        cls = LogRegDriver(temp, REFERENCE_DATA)
    elif model_choice == 'SGD':
        cls = SGDDriver(temp, REFERENCE_DATA)
    elif model_choice == 'RandomForest':
        cls = RandomForestDriver(temp, REFERENCE_DATA)
    elif model_choice == 'ExtraTrees':
        cls = ExtraTreesDriver(temp, REFERENCE_DATA)
    elif model_choice == 'Lasso':
        cls = LassoDriver(temp, REFERENCE_DATA)
    elif model_choice == 'ElasticNet':
        cls = ElasticNetDriver(temp, REFERENCE_DATA)
    elif model_choice == 'Ridge':
        cls = RidgeDriver(temp, REFERENCE_DATA)
    elif model_choice == 'LinearSVC':
        cls = LinearSVCDriver(temp, REFERENCE_DATA)
    elif model_choice == 'RBFSVR':
        cls = RBFSVRDriver(temp, REFERENCE_DATA)
    elif model_choice == 'GB':
        cls = GBDriver(temp, REFERENCE_DATA)
    else:
        print """
            model choice not recognized. Options are:
            Regression    RandomForest    Lasso
            ElasticNet    Ridge           LinearSVR    
            RBFSVR        GB
            """
        sys.exit()
    cls.classify()
    return cls.toKaggle()


def analysis(foldername, outdir, referencenum, model_choice):
    """
    Start the analysis

    Input:
        1) Path to the driver directory
        2) Path where the submission file should be written
        3) Number of drivers to compare against
        4) Choice of particular classification/regression model
    """
    version_number = re.sub("[^0-9.]", "", __version__)
    seed()
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
    referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
    referencedrivers = []
    for referencefolder in referencefolders:
        referencedrivers.append(Driver(referencefolder))
    generatedata(referencedrivers)
    results = Parallel(n_jobs=4)(delayed(perform_analysis)(folder, model_choice) for folder in folders)
    with open(os.path.join(outdir, "py"+str(model_choice)+"_{0}_v".format(submission_id)+version_number+".csv"), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for item in results:
            writefile.write("%s\n" % item)
    print 'Model chosen was %s' % model_choice 
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    analysis(os.path.join(MyPath, "..", "driversnew"), MyPath, 100, sys.argv[1])
