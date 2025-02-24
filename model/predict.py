import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import keras


def input_data(ticker):
    df_with_ticker = pd.read_csv('data/df_with_ticker.csv')
    df = pd.read_csv('data/data.csv')
    X_sc = joblib.load('model/X_scaler.pkl')
    idx = df_with_ticker['Ticker'] == ticker
    all_data = df_with_ticker[idx]
    quarter = all_data[2:]['Quarter']
    quarter = pd.concat([quarter, pd.Series(['24.Q4', '25.Q1'])])
    all_data = df[idx]
    data = all_data[2:]
    data_ar = np.array(data)
    data_sc = X_sc.transform(data_ar)
    
    return data, data_ar, data_sc, quarter

def predictor(ticker):
    model = keras.saving.load_model("model/lstm_model.keras")
    y_sc = joblib.load('model/y_scaler.pkl')
    data, data_ar, data_sc , quarter = input_data(ticker)
    pred = model.predict(data_sc.reshape(1, 23, 28))
    y_pred_original = y_sc.inverse_transform(pred.reshape(-1, 1))
    y_pred_original = np.exp(y_pred_original[0])
    real = data['close']
    real = np.array(real)
    values = np.hstack([real, (real[-1] + y_pred_original) / 2, y_pred_original])
    plt.figure(figsize=(14, 4))
    plt.plot(quarter, values, label = 'real')
    plt.plot(real, color = 'red')
    plt.grid()
    plt.text(24, y_pred_original + 3, y_pred_original[0], fontsize=10, ha='center', color='black')
    plt.title(ticker)
    plt.legend()
    return y_pred_original, plt