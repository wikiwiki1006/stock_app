import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib

df = pd.read_csv('data/data.csv')

def make_X_data(): # data.csv 입력
    lis = []
    for i in range(df.shape[0] // 25):
        lis.append((i*25 + 24))
        lis.append((i*25 + 23))
    df_filtered = df.drop(lis)
    X = np.array(df_filtered)
    X = X.reshape(-1, 23, 28)

    return X

def make_y_data():
    y = []
    for i in range(df.shape[0] // 25):
        close = df.loc[i*25 + 24]['close']
        y.append(close)
    y = np.array(y)
    y = y.reshape(-1, 1)
    y = np.log(y)

    return y

def process_data():
    X = make_X_data()
    y = make_y_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state= 1)
    X_sc = StandardScaler()
    y_sc = StandardScaler()

    X_train_scaled = X_sc.fit_transform(X_train.reshape(-1, 28))
    y_train_scaled = y_sc.fit_transform(y_train)
    X_test_scaled = X_sc.transform(X_test.reshape(-1, 28))
    y_test_scaled = y_sc.transform(y_test)

    joblib.dump(X_sc, 'model/X_scaler.pkl')
    joblib.dump(y_sc, 'model/y_scaler.pkl')
    return X_train_scaled, y_train_scaled


def lstm_model():
    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences = True, input_shape=(23, 28)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, return_sequences = False),
    tf.keras.layers.Dense(30, activation = 'relu'),
    tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_model():
    model = lstm_model()
    X_train_scaled, y_train_scaled = process_data()
    model.fit(X_train_scaled.reshape(-1, 23, 28), y_train_scaled, batch_size= 5 ,epochs=100, validation_split=0.1)
    model.save('model/lstm_model.keras')

train_model()