import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__)
#This application is refer in python.config while in WSGIPath
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            lunch=request.form.get('lunch'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            reading_score=request.form.get('reading_score'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=request.form.get('writing_score'),
        )
        pred_df=data.get_data_as_df()
        predict=PredictPipeline()
        res=predict.predict(pred_df)
        return render_template("home.html",results=res[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)