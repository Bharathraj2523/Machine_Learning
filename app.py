from flask import Flask,request,render_template  
import numpy as np 
import pandas as pd 
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictionPipeline
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            temperature=float(request.form.get('temperature')),
            irradiance=float(request.form.get('irradiance')),
            humidity=float(request.form.get('humidity')),
            panel_age=float(request.form.get('panel_age')),
            maintenance_count=int(float(request.form.get('maintenance_count'))),   # safer
            soiling_ratio=float(request.form.get('soiling_ratio')),
            voltage=float(request.form.get('voltage')),
            current=float(request.form.get('current')),
            module_temperature=float(request.form.get('module_temperature')),
            cloud_coverage=float(request.form.get('cloud_coverage')),
            wind_speed=float(request.form.get('wind_speed')),
            pressure=float(request.form.get('pressure')),
            installation_type_encoded=int(float(request.form.get('installation_type_encoded'))),  # just in case
            error_code_E00=int(float(request.form.get('error_code_E00'))),
            error_code_E01=int(float(request.form.get('error_code_E01'))),
            error_code_E02=int(float(request.form.get('error_code_E02'))),
            error_code_nan=int(float(request.form.get('error_code_nan'))),
            area=float(request.form.get('area')),
            power=float(request.form.get('power'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
