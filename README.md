# Flight Delay Prediction System
Predict delays. Estimate duration. Built with ML + Flask.

A Machine Learning system that predicts whether a flight will be **delayed or on-time**, and estimates the **expected delay duration** — based on airline, departure time, and weather conditions.

---

## Features

### Real-time Prediction
Enter flight details in a web form → get instant prediction with confidence %

### Delay Duration Estimation
Not just "Delayed" — predicts how many **hours/minutes** the delay will be (MAE ~10 mins)

### Smart Ranking (Model)
- **Random Forest Classifier** — 65% accuracy, 85% recall for delayed flights
- **Random Forest Regressor** — estimates delay minutes from airline, time, weather

### Interactive Visualizations (5 Plots)
| Plot | What it Shows |
|------|--------------|
| `eda_plots.png` | Flight status distribution, delays by airline, hour, day of week |
| `weather_vs_delay.png` | Temperature, visibility, wind speed split by delay status |
| `confusion_matrix.png` | Model prediction accuracy breakdown |
| `feature_importance.png` | Which factors matter most (visibility, hour, airline, etc.) |
| `roc_curve.png` | Model quality score (AUC) |

### Web App UI
- Form with airline, date/time, temperature, visibility, wind speed
- Color-coded results (green = on-time, red = delayed)
- Clean responsive design with gradient theme

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3 + Flask 3 |
| **ML** | scikit-learn (Random Forest) |
| **Frontend** | HTML/CSS + Jinja2 Templates |
| **Data** | Synthetic (1000 flights with realistic patterns) |
| **Model Persistence** | Joblib (.pkl) |
| **Visualization** | Matplotlib + Seaborn |

---

## Project Structure

```
flight-delay-system/
│
├── flight_delay_ml.py          # ML training + data generation + plots
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies
├── README.md
│
├── templates/
│   └── index.html              # Web form UI
│
├── flight_delay_model.pkl      # Trained classifier
├── flight_delay_reg_model.pkl  # Trained regressor (delay mins)
├── model_features.pkl          # Feature column names
├── flights_data.csv            # Generated dataset
│
├── eda_plots.png               # EDA visualizations
├── weather_vs_delay.png        # Weather impact viz
├── confusion_matrix.png        # Model confusion matrix
├── feature_importance.png      # Feature importance chart
└── roc_curve.png              # ROC curve
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional — models already included)
```bash
python flight_delay_ml.py
```
- Generates `flights_data.csv` with 1000 synthetic flights
- Trains classifier + regressor
- Saves 5 visualization PNGs

### 3. Start Web App
```bash
python app.py
```

### 4. Open Browser
**http://127.0.0.1:5000**

---

## How It Works

### Data Generation (Realistic Patterns)
Delay probability depends on multiple factors:

| Factor | Impact |
|--------|--------|
| **Airline** | JetBlue = 50% delay, Southwest = 25% delay |
| **Time of Day** | Late evening = 1.5x multiplier, Morning = 0.7x |
| **Visibility** | ≤5km → +25% delay probability |
| **Wind Speed** | >20 km/h → +20% delay probability |
| **Temperature** | <0°C or >35°C → +10% delay probability |

Delay minutes (15-180 mins) are calculated from the same factors with airline-specific adjustments.

### ML Pipeline
```
Raw Data → Feature Engineering → Train/Test Split → Random Forest → Prediction
```

Features used:
- `HourOfDay`, `DayOfWeek`, `Temperature(C)`, `Visibility(km)`, `WindSpeed(km/h)`
- One-hot encoded `Airline` (Delta, United, American, Southwest, JetBlue)

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Show prediction form |
| `/` | POST | Submit flight data → get prediction |

---

## Retraining with Custom Data

Edit `flight_delay_ml.py`:
- Change `num_flights` for larger dataset
- Modify `AIRLINE_DELAY_PROB` / `AIRLINE_DELAY_MINS` for different patterns
- Add new features in the data generation loop

Then run:
```bash
python flight_delay_ml.py
```

---


### Web App
```
┌─────────────────────────────────┐
│  Flight Delay Prediction        │
│  Enter flight details...        │
│                                 │
│  Airline:    [JetBlue     ▼]    │
│  Date:       [2026-07-19   ]    │
│  Time:       [22:30        ]    │
│  Temp:       [12           ] °C │
│  Visibility: [5 km        ▼]    │
│  Wind:       [28          ] km/h│
│                                 │
│  ┌───────────────────────────┐  │
│  │     Prediction Result     │  │
│  │        DELAYED            │  │
│  │         80.3%             │  │
│  │       confidence          │  │
│  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │  │
│  │ Expected Delay            │  │
│  │       1h 39m              │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

---
