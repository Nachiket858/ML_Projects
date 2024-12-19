from flask import Flask,request , jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

## import ridge and scalar pkl file   loding models
ridge_model = pickle.load(open("models\\RidgeRegressor.pkl",'rb'))
scalar_model = pickle.load(open("models\\Scalar.pkl",'rb'))

## Route for home page
# @app.route('/home')
# def index():
#     return render_template('index.html')


@app.route("/",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        temperature = request.form.get('Temperature', type=float)
        rh = request.form.get('RH', type=float)
        ws = request.form.get('Ws', type=float)
        rain = request.form.get('Rain', type=float)
        ffmc = request.form.get('FFMC', type=float)
        dmc = request.form.get('DMC', type=float)
        isi = request.form.get('ISI', type=float)
        classes = request.form.get('Classes', type=float)
        region = request.form.get('Region', type=float)

        data=scalar_model.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        print(data)
        result = ridge_model.predict(data)
        print(result)
    # Return the result as a response
        return render_template('home.html',result=result[0])
    
    else:
        return render_template ("home.html")




if __name__ == '__main__':
    app.run()