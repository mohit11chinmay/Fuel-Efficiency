FuelIQ — Smart Fuel-Efficiency Predictor
A Flask-based ML API that predicts vehicle fuel efficiency, emissions, and cost-of-ownership metrics using a Gradient Boosting Regressor.

🚀 Quick Start
bash

Copy
# Run the API server
python3 app.py
The API starts at http://localhost:8080.

📋 Requirements
Flask
flask-cors
numpy
pandas
scikit-learn
Install them with:

bash

Copy
pip install flask flask-cors numpy pandas scikit-learn
🧪 API Endpoints
GET /health
Returns the server status and trained model metrics.

Response:

json

Copy
{
  "status": "ok",
  "model": "GradientBoostingRegressor",
  "metrics": {
    "r2": 0.96,
    "rmse": 1.45,
    "mae": 1.12,
    "cv_mean": 0.96,
    "cv_std": 0.01,
    "train_samples": 8000,
    "test_samples": 2000
  }
}
POST /predict
Predicts fuel efficiency and related metrics for a given vehicle configuration.

Request body (JSON):

vehicle_class
string
compact, midsize, fullsize, suv, truck, van, sports
fuel_type
string
gasoline, diesel, hybrid, electric
transmission
string
auto, manual, cvt, dct
drive_type
string
fwd, rwd, awd, 4wd
model_year
int
2015–2024
displacement
float
Engine displacement in liters
cylinders
int
Number of cylinders
horsepower
int
Engine horsepower
curb_weight
int
Vehicle weight in lbs
Example request:

json

Copy
{
  "vehicle_class": "compact",
  "fuel_type": "hybrid",
  "transmission": "cvt",
  "drive_type": "fwd",
  "model_year": 2023,
  "displacement": 1.8,
  "cylinders": 4,
  "horsepower": 140,
  "curb_weight": 3000
}
Response:

json

Copy
{
  "mpg_combined": 42.3,
  "city_mpg": 34.7,
  "highway_mpg": 48.2,
  "annual_fuel_cost": 1238,
  "range_miles": 592,
  "co2_gkm": 105.2,
  "co2_annual_tons": 2.86,
  "trees_to_offset": 130,
  "rating": "EXCELLENT",
  "rating_class": "excellent",
  "feature_importances": [
    { "name": "curb_weight", "importance": 0.35 },
    { "name": "horsepower", "importance": 0.28 },
    ...
  ],
  "model_metrics": { ... }
}
📊 Model Details
Algorithm
Gradient Boosting Regressor
Estimators
300
Max Depth
5
Learning Rate
0.08
Training Data
10,000 synthetically generated vehicles (8k train / 2k test)
Features
9 (vehicle class, fuel type, transmission, drive type, model year, displacement, cylinders, horsepower, curb weight)
Evaluation
5-fold cross-validation (R² scoring)
Feature Importance
The model ranks features by importance. Typically, curb weight and horsepower are the strongest predictors of fuel efficiency.

Ratings
≥ 40
EXCELLENT
≥ 30
GOOD
≥ 22
AVERAGE
< 22
POOR
🧮 Derived Metrics
City MPG = Combined × 0.82
Highway MPG = Combined × 1.14
Annual Fuel Cost = (12,000 mi / MPG) × $3.55/gal
Range (miles) = Combined MPG × Tank Size (14 gal standard, 26 gal for trucks)
CO₂ (g/km) = 8887 / (MPG × 1.60934) — zero for electric
CO₂ Annual (tons) = (12,000 / MPG) × 8.887 kg / 1000 — zero for electric
Trees to Offset = Annual CO₂ tons / 0.022 tons per tree per year
🏗️ Project Structure
Fuel-Efficiency/
├── app.py          # Flask API + ML model (all-in-one)
├── index.html      # Frontend UI
├── README.md       # This file
└── requirements.txt (optional)
📝 Notes
The model trains on a synthetically generated dataset (~10k vehicles) using realistic distributions for vehicle class, fuel type, transmission, drive type, and engine specs.
Categorical features are encoded with sklearn.LabelEncoder.
CORS is enabled so the frontend can call the API from a different origin.
📄 License
This project is open-source.

Author: mohit11chinmay
