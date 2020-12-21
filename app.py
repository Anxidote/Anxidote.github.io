import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('quiz.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = request.form.values()
    
    prediction = Predictor1.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('surveyResult.html', prediction_text='Predictions are $ {}'.format(output))

