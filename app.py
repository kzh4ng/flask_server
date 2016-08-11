#!flask/bin/python
from flask import Flask, redirect, url_for, request, jsonify, render_template
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import json
import ridge_model

app = Flask(__name__)
initialization = False

@app.route('/success/<hour>/<date>/<weekday>/<amount>')
def success(hour,date,weekday,amount):
    hr = int(hour)
    doy = int(date)
    wd = int(weekday)
    amt = int(amount)
    predictions = []
    poly = ridge_model.poly

    for i in range(0, amt*24):
        dictionary = {}
        features = [hr, wd, doy]
        np_features = np.array(features)
        polynomial_features = poly.fit_transform(np_features.reshape(1,-1))
        dictionary['hr'] = hr
        dictionary['doy'] = doy
        dictionary['wd'] = wd
        dictionary['prediction'] = model.predict(polynomial_features)[0]
        predictions.append(dictionary)
        hr += 1
        if hr == 24:
            hr = 0
            wd = (wd+1)%7
            doy = (doy+1)%365
    return json.dumps(predictions)

@app.route('/login',methods = ['POST'])
def login():
    date_form = request.form['DOY']
    hour_form = request.form['HR']
    day_form = request.form['WEEK_DAY']
    amount_form =request.form['AMOUNT']
    return redirect(url_for('success', hour = hour_form, weekday = day_form, date = date_form, amount = amount_form))

@app.route('/')
def index():
    build_model()
    return render_template('login.html', title='Home')

def build_model():
    global initialization
    global model
    if initialization == False:
        df = ridge_model.preprocessing()
        model = ridge_model.regression(df)
        initialization = True

def format_date(date): #MM/DD/YYYY -> MMDD
    date = date.replace("/","")
    date = date[:-4]
    return date

if __name__ == '__main__':
   app.run(debug = True)
   print("hello")