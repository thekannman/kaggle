import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    df1_1 = df.drop('target',1)
    df1_2 = df['target']
    df2 = pd.read_csv('rf_train.csv')
    df2 = df2.drop('id',1)
    df_list = [df1_1,df2,df1_2]
    df = pd.concat(df_list,axis=1)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    df2 = pd.read_csv('rf_test.csv')
    df2 = df2.drop('id',1)
    df_list = [df,df2]
    df = pd.concat(df_list,axis=1)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(int).astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

X, y, encoder, scaler = load_train_data('train.csv')
X_test, ids = load_test_data('test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dropout0', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout2', DropoutLayer),
           ('dense2', DenseLayer),
           ('dropout3', DropoutLayer),
           ('dense3', DenseLayer),
           ('dropout4', DropoutLayer),
           ('dense4', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropout0_p=0.10,
                 dense0_num_units=500,
                 dropout1_p=0.5,
                 dense1_num_units=1000,
                 dropout2_p=0.5,
                 dense2_num_units=500,
                 dropout3_p=0.5,
                 dense3_num_units=500,
                 dropout4_p=0.5,
                 dense4_num_units=500,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=500)

net0.fit(X, y)

make_submission(net0, X_test, ids, encoder)

