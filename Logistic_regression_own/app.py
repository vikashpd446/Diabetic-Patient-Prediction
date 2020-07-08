from flask import Flask, render_template, request
import jsonify
import pandas
import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)
model = pickle.load(open('modelForPrediction.pickle','rb'))
scaler = pickle.load(open('scaler.pickle','rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

standard_scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        try:
            Pregnancies = int(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])
            trans = ([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
            Scaler_data = scaler.transform(trans)
            prediction = model.predict(Scaler_data)
            if prediction==1:
                return render_template('index.html',prediction_text="Diabetic")
            else:
                return render_template('index.html',prediction_text="Non-Diabetic")
            
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')
    
if __name__=="__main__":
    app.run(port=7000,debug=True)
            
            
            
            
            
            
            