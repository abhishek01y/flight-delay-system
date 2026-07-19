# pip install pandas scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

random.seed(42)
np.random.seed(42)

AIRLINE_DELAY_PROB = {
    "Delta": 0.30, "United": 0.45, "American": 0.40,
    "Southwest": 0.25, "JetBlue": 0.50
}

AIRLINE_DELAY_MINS = {
    "Delta": 0, "United": 10, "American": 5,
    "Southwest": -5, "JetBlue": 15
}

def get_hour_delay_multiplier(hour):
    if 6 <= hour < 12:    return 0.7
    if 12 <= hour < 18:   return 0.9
    if 18 <= hour < 23:   return 1.5
    return 1.3

def get_weather_delay_boost(temp, visibility, wind):
    boost = 0
    if visibility <= 5:    boost += 0.25
    elif visibility <= 10: boost += 0.10
    if wind > 20:          boost += 0.20
    elif wind > 12:        boost += 0.10
    if temp < 0 or temp > 35: boost += 0.10
    return boost

def generate_delay_minutes(airline, hour, visibility, wind):
    base = random.randint(15, 45)
    airline_adj = AIRLINE_DELAY_MINS[airline]
    if hour >= 18 or hour < 6:
        time_adj = random.randint(10, 30)
    else:
        time_adj = 0
    weather_adj = 0
    if visibility <= 5:    weather_adj += random.randint(15, 30)
    elif visibility <= 10: weather_adj += random.randint(5, 15)
    if wind > 20:          weather_adj += random.randint(10, 25)
    elif wind > 12:        weather_adj += random.randint(5, 10)
    return max(base + airline_adj + time_adj + weather_adj, 5)

def generate_flight_status(airline, hour, temp, visibility, wind):
    base_prob = AIRLINE_DELAY_PROB[airline]
    hour_mult = get_hour_delay_multiplier(hour)
    weather_boost = get_weather_delay_boost(temp, visibility, wind)
    prob = min(base_prob * hour_mult + weather_boost, 0.95)
    return "Delayed" if random.random() < prob else "On-time"

num_flights = 1000
flights = []

print("Generating realistic flight data...")
for i in range(num_flights):
    flight_number = f"FL{random.randint(1000, 9999)}"
    airline = random.choice(list(AIRLINE_DELAY_PROB.keys()))
    dep_hour = random.randint(0, 23)
    dep_day = random.randint(0, 30)
    departure_time = datetime.now() + timedelta(days=dep_day, hours=dep_hour, minutes=random.randint(0, 59))
    flight_duration = random.randint(1, 6)
    arrival_time = departure_time + timedelta(hours=flight_duration)
    temperature = random.randint(-5, 42)
    visibility = random.choice([5, 10, 15, 20, 25])
    wind_speed = random.randint(0, 35)
    flight_status = generate_flight_status(airline, dep_hour, temperature, visibility, wind_speed)
    delay_minutes = generate_delay_minutes(airline, dep_hour, visibility, wind_speed) if flight_status == "Delayed" else 0

    flights.append({
        "FlightNumber": flight_number, "Airline": airline,
        "DepartureTime": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "ArrivalTime": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
        "FlightStatus": flight_status,
        "DelayMinutes": delay_minutes,
        "Temperature(C)": temperature, "Visibility(km)": visibility,
        "WindSpeed(km/h)": wind_speed
    })

df = pd.DataFrame(flights)
df.to_csv("flights_data.csv", index=False)

delay_count = (df['FlightStatus'] == 'Delayed').sum()
avg_delay = df[df['FlightStatus'] == 'Delayed']['DelayMinutes'].mean()
print(f"Generated {num_flights} flights ({delay_count} delayed, {num_flights - delay_count} on-time)")
print(f"Average delay: {avg_delay:.0f} minutes\n")

df['DepartureTime'] = pd.to_datetime(df['DepartureTime'])
df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'])
df['HourOfDay'] = df['DepartureTime'].dt.hour
df['DayOfWeek'] = df['DepartureTime'].dt.dayofweek

df_eda = df.copy()

df = pd.get_dummies(df, columns=['Airline'], drop_first=True)
df['FlightStatus'] = (df['FlightStatus'] == 'Delayed').astype(int)
df.drop(columns=['FlightNumber', 'DepartureTime', 'ArrivalTime'], inplace=True)

X = df.drop(columns=['FlightStatus', 'DelayMinutes'])
y_cls = df['FlightStatus']

X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)

cls_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
cls_model.fit(X_train, y_cls_train)

y_pred = cls_model.predict(X_test)
y_prob = cls_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_cls_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_cls_test, y_pred, target_names=["On-time", "Delayed"], output_dict=True)
print(f"\n{'Class':<10}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}")
for class_name in ["On-time", "Delayed"]:
    print(f"{class_name:<10}{report[class_name]['precision']:<12.2f}{report[class_name]['recall']:<10.2f}{report[class_name]['f1-score']:<10.2f}")

df_delayed = df[df['FlightStatus'] == 1].copy()
if len(df_delayed) > 50:
    X_reg = df_delayed.drop(columns=['FlightStatus', 'DelayMinutes'])
    y_reg = df_delayed['DelayMinutes']

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    reg_model.fit(X_reg_train, y_reg_train)

    y_reg_pred = reg_model.predict(X_reg_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    print(f"\nDelay Minutes Regression MAE: {mae:.1f} mins")
else:
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train[:10], y_cls_train[:10] * 30)

joblib.dump(cls_model, "flight_delay_model.pkl")
joblib.dump(reg_model, "flight_delay_reg_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("Models saved: flight_delay_model.pkl (classifier), flight_delay_reg_model.pkl (regressor)")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight='bold')

df_eda['FlightStatus'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'],
    ax=axes[0, 0], title='Flight Status Distribution'
)
axes[0, 0].set_ylabel('')

df_eda.groupby('Airline')['FlightStatus'].apply(
    lambda x: (x == 'Delayed').sum()
).sort_values().plot(kind='barh', color='salmon', ax=axes[0, 1], title='Delays by Airline')
axes[0, 1].set_xlabel('Number of Delays')

hourly_delays = df_eda.groupby('HourOfDay')['FlightStatus'].apply(
    lambda x: (x == 'Delayed').sum()
)
hourly_delays.plot(kind='bar', color='skyblue', ax=axes[1, 0], title='Delays by Hour of Day')
axes[1, 0].set_ylabel('Number of Delays')

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_delays = df_eda.groupby('DayOfWeek')['FlightStatus'].apply(
    lambda x: (x == 'Delayed').sum()
)
daily_delays.plot(kind='bar', color='lightgreen', ax=axes[1, 1], title='Delays by Day of Week')
axes[1, 1].set_ylabel('Number of Delays')
axes[1, 1].set_xticklabels(day_names)

plt.tight_layout()
plt.savefig("eda_plots.png")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Weather Conditions vs Flight Status", fontsize=16, fontweight='bold')
colors = {'On-time': 'blue', 'Delayed': 'red'}
for ax, col, xlabel in zip(axes, ['Temperature(C)', 'Visibility(km)', 'WindSpeed(km/h)'],
                            ['Temperature (°C)', 'Visibility (km)', 'Wind Speed (km/h)']):
    for status in ['On-time', 'Delayed']:
        subset = df_eda[df_eda['FlightStatus'] == status]
        ax.hist(subset[col], alpha=0.6, label=status, color=colors[status], bins=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.legend()
plt.tight_layout()
plt.savefig("weather_vs_delay.png")
plt.close()

conf_matrix = confusion_matrix(y_cls_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["On-time", "Delayed"], yticklabels=["On-time", "Delayed"])
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': cls_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
colors_bar = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_importance)))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_bar)
plt.xlabel('Importance Score')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for bar, val in zip(bars, feature_importance['Importance']):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

fpr, tpr, _ = roc_curve(y_cls_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

print(f"\nPlots saved: eda_plots.png, weather_vs_delay.png, confusion_matrix.png, feature_importance.png, roc_curve.png")
