import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax,rectify,LeakyRectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    df1_1 = df.drop('units',1)
    df1_2 = df['units']
    df_list = [df1_1,df1_2]
    df = pd.concat(df_list,axis=1)
    X = df.values.copy()
    np.random.shuffle(X)
    X, y = X[:, 0:-1].astype(np.float32), X[:, -1]
    y = np.log(y+1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    id_df = df['store_nbr'].astype(str) + '_' + df['item_nbr'].astype(str) + '_' + df['date'].astype(str)
    df = df.drop('date',1)
    ids = id_df.values.copy()
    X = df.values.copy()
    X = X.astype(np.float32)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, name='my_neural_net_submission.csv'):
    y_pred = clf.predict(X_test)
    y_pred = np.exp(y_pred)-1
    with open(name, 'w') as f:
        f.write('id,units')
        f.write('\n')
        for id, pred in zip(ids, y_pred):
            line = id + ',' + pred[0].astype(str)
            f.write(line)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

X, y, scaler = load_train_data('train_weather.csv')
X = X.astype(np.float32)
y = y.astype(np.float32)
y = y.reshape((-1,1))

X_test, ids = load_test_data('test_weather.csv', scaler)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           #('dropout0', aaussianNoiseLayer)yers=layers0,
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense1', DenseLayer),
           #('dropout2', DropoutLayer),
           #('dense2', DenseLayer),
           #('dropout3', DropoutLayer),
           #('dense3', DenseLayer),
           #('dropout4', DropoutLayer),
           #('dense4', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 #dropout0_p=0.1,
                 dense0_num_units=200,
                 dropout1_p=0.5,
                 dense1_num_units=200,
                 #dropout2_p=0.5,
                 #dense2_num_units=500,
                 #dropout3_p=0.5,
                 #dense3_num_units=500,
                 #dropout4_p=0.5,
                 #dense4_num_units=500,
                 output_num_units=1,
                 output_nonlinearity=None,

                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,

                 regression=True,
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=40)

net0.fit(X, y)

make_submission(net0, X_test, ids)

