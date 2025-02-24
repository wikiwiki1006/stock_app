from flask import Flask, render_template, request
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

import keras

from model.predict import predictor


app = Flask(__name__)

# 모델 로드 (예시)
model = keras.saving.load_model('model/lstm_model.keras')

@app.route('/')
def index():
    return render_template('index.html')  # HTML 페이지 렌더링

@app.route('/predict', methods=['POST'])
def prediction():
    
    ticker = request.form['ticker'] 


    # 예측 작업 (여기서 예측 함수 호출)
    predicted_value, plt = predictor(ticker)

    # 그래프 파일로 저장
    graph_path = 'static/graph.png'
    plt.savefig(graph_path)
    plt.close()  # 그래프 종료

    return render_template('index.html', prediction=predicted_value[0], graph_url=graph_path) 


if __name__ == '__main__':
    app.run()