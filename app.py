import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load("flight_delay_model.pkl")
reg_model = joblib.load("flight_delay_reg_model.pkl")
feature_columns = joblib.load("model_features.pkl")

AIRLINES = ["Delta", "United", "American", "Southwest", "JetBlue"]

def prepare_input(airline, dep_datetime, temperature, visibility, wind_speed):
    hour = dep_datetime.hour
    dayofweek = dep_datetime.weekday()

    data = {
        "HourOfDay": hour,
        "DayOfWeek": dayofweek,
        "Temperature(C)": float(temperature),
        "Visibility(km)": float(visibility),
        "WindSpeed(km/h)": float(wind_speed),
    }

    for col in feature_columns:
        if col.startswith("Airline_"):
            data[col] = 1 if col == f"Airline_{airline}" else 0

    df = pd.DataFrame([data])
    df = df[feature_columns]
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    result_class = None
    delay_mins = None
    delay_str = None
    input_data = {}

    if request.method == "POST":
        airline = request.form["airline"]
        dep_date = request.form["departure_date"]
        dep_time = request.form["departure_time"]
        temperature = request.form["temperature"]
        visibility = request.form["visibility"]
        wind_speed = request.form["wind_speed"]

        input_data = {
            "airline": airline, "departure_date": dep_date,
            "departure_time": dep_time, "temperature": temperature,
            "visibility": visibility, "wind_speed": wind_speed
        }

        dep_datetime = datetime.strptime(f"{dep_date} {dep_time}", "%Y-%m-%d %H:%M")
        X = prepare_input(airline, dep_datetime, temperature, visibility, wind_speed)

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if pred == 1:
            prediction = "Delayed"
            probability = round(prob[1] * 100, 1)
            result_class = "delayed"
            delay_mins = int(round(reg_model.predict(X)[0]))
            hours = delay_mins // 60
            mins = delay_mins % 60
            if hours > 0:
                delay_str = f"{hours}h {mins}m"
            else:
                delay_str = f"{mins} min"
        else:
            prediction = "On-time"
            probability = round(prob[0] * 100, 1)
            result_class = "ontime"

    return render_template("index.html", prediction=prediction, probability=probability,
                           result_class=result_class, delay_mins=delay_mins, delay_str=delay_str,
                           input_data=input_data, airlines=AIRLINES)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)
