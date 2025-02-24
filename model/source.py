import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import joblib

raw_df = pd.read_csv('data/final_data.csv')
df = raw_df.drop(columns=['Quarter', 'Sector','Ticker'])



def reverse_data():
    chunks = [raw_df.iloc[i:i + 25] for i in range(0, raw_df.shape[0], 25)]
    reversed_chunks = [chunk[::-1] for chunk in chunks]

    reversed_df = pd.concat(reversed_chunks, ignore_index=True)
    df_with_ticker = pd.DataFrame(reversed_df)

    return df_with_ticker


def make_X_data():
    lis = []
    for i in range(df.shape[0] // 25):
        lis.append((i*25 + 24))
        lis.append((i*25 + 23))
    df_filtered = df.drop(lis)
    X = np.array(df_filtered)
    X = X.reshape(418, 23, 28)

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


def process_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state= 1)
    X_sc = StandardScaler()
    y_sc = StandardScaler()

    X_train_scaled = X_sc.fit_transform(X_train.reshape(-1, 28))
    y_train_scaled = y_sc.fit_transform(y_train)
    X_test_scaled = X_sc.transform(X_test.reshape(-1, 28))
    y_test_scaled = y_sc.transform(y_test)

    joblib.dump(X_sc, 'X_scaler.pkl')
    joblib.dump(y_sc, 'y_scaler.pkl')

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_sc, y_sc

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

def train_model(model, X_train_scaled, y_train_scaled):
    history = model.fit(X_train_scaled.reshape(-1, 23, 28), y_train_scaled, batch_size= 5 ,epochs=50, validation_split=0.2)
    model.save('my_model.h5')
    # plt.plot(history.history['loss'], label='Train Loss (MSE)')
    # plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    # plt.title('Model Training Loss (MSE)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


def tester(model, X_test_scaled, y_test_scaled, y_sc):
    y_pred = model.predict(X_test_scaled.reshape(-1, 23, 28))
    # 예측값 역정규화 (스케일링을 되돌림)
    y_pred_original = y_sc.inverse_transform(y_pred)
    y_test_original = y_sc.inverse_transform(y_test_scaled.reshape(-1, 1))
    y_pred_original = np.exp(y_pred_original)
    y_test_original = np.exp(y_test_original)

    odd = y_pred_original - y_test_original
    error = abs(odd) / y_test_original
    error_ratio = sum(error) / y_test_original.shape[0]
    error_ratio = error_ratio * 100

    # print(f'오차율 : {error_ratio}')
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test_original, label='True Prices')
    # plt.plot(y_pred_original, label='Predicted Prices')
    # plt.legend()
    # plt.show()


df_with_ticker =  reverse_data()
X = make_X_data()
y = make_y_data()
X_train_scaled, y_train_scaled, \
X_test_scaled, y_test_scaled, X_sc, y_sc \
= process_data(X, y)
model = lstm_model()
train_model(model, X_train_scaled, y_train_scaled)
tester(model, X_test_scaled, y_test_scaled, y_sc)



