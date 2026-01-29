from flask import Flask,request,jsonify
import joblib
import pandas as pd

app = Flask(__name__)

#Load Trained models
reg_model = joblib.load("best_rf_regression_pipeline.pkl")
jam_model = joblib.load("best_jam_classifier_pipeline.pkl")

#Helper:Build input DataFrame

def build_input(data):
    df = pd.DataFrame([{
        'lag_1':data['lag_1'],
        'lag_6':data['lag_6'],
        'lag_12':data['lag_12'],
        'rolling_mean_6':data['rolling_mean_6'],#average of speeds
        'rolling_std_6':data['rolling_std_6'],#speed high std= 1--2
        'hour':data['hour'],                  #speed low std = 8--12
        'day':data['day']
    }])
    return df

#Root Endpoint
@app.route("/",methods=["GET"])
def home():
    return jsonify({
        "message":"Smart Traffic ML Flask API is running"
    })


@app.route("/predict_all",methods= ["POST"])
def predict_all():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form
    X = build_input(data)

    #Speed prediction
    speed_pred = reg_model.predict(X)[0]

    #Jam prediction
    jam_class = jam_model.predict(X)[0]

    label_map = {0:"Jam",1:"Slow",2:"Free"}

    return jsonify({
        "Predicted_Speed:":float(speed_pred),
        "Traffic_condition":label_map[int(jam_class)]
    })



#Run Flask App

if __name__=="__main__":
 app.run(debug=True)