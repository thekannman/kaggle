import inspect
import os
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "/home/jiwei/kaggle/tradetext/xgboost-master/wrapper")

sys.path.append(code_path)
import xgboost as xgb
import numpy as np
class xgb_classifier:
    def __init__(self,eta,min_child_weight,depth,num_round,threads=8,boost_from_exist_prediction=False,exist_num_round=20):
        self.eta=eta
        self.min_child_weight=min_child_weight
        self.depth=depth
        self.num_round=num_round
        self.boost_from_exist_prediction=boost_from_exist_prediction
        self.exist_num_round=exist_num_round  
        self.threads=threads
       
    def train_predict(self,X_train,y_train,X_test,base_train_prediction,base_test_prediction):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'binary:logistic'

        param['bst:eta'] = self.eta
        param['colsample_bytree']=1
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['eval_metric'] = 'auc'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round

        xgmat_test = xgb.DMatrix(X_test,missing=-999)
    
        if self.boost_from_exist_prediction:
        # train xgb with existing predictions
        # see more at https://github.com/tqchen/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
       
            xgmat_train.set_base_margin(base_train_prediction)
            xgmat_test.set_base_margin(base_test_prediction)
            bst = xgb.train(param, xgmat_train, self.exist_num_round, watchlist )
        else:
            bst = xgb.train( plst, xgmat_train, num_round, watchlist )
        ypred = bst.predict(xgmat_test)
        return ypred
        



