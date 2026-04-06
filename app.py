"""
FuelIQ — Flask API Backend
===========================
Run: python3 app.py
API runs at: http://localhost:8080
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

app = Flask(__name__)
CORS(app)

MODEL = None
PREPROCESSOR = None
FEATURE_COLS = None
MODEL_METRICS = {}


def generate_dataset(n_samples=10000):
    vehicle_classes = ['compact', 'midsize', 'fullsize', 'suv', 'truck', 'van', 'sports']
    fuel_types      = ['gasoline', 'diesel', 'hybrid', 'electric']
    transmissions   = ['auto', 'manual', 'cvt', 'dct']
    drive_types     = ['fwd', 'rwd', 'awd', '4wd']

    class_weights = [0.25, 0.25, 0.15, 0.20, 0.08, 0.04, 0.03]
    fuel_weights  = [0.65, 0.15, 0.15, 0.05]
    trans_weights = [0.55, 0.15, 0.20, 0.10]
    drive_weights = [0.45, 0.20, 0.25, 0.10]

    data = []
    for _ in range(n_samples):
        v_class = np.random.choice(vehicle_classes, p=class_weights)
        fuel    = np.random.choice(fuel_types,      p=fuel_weights)
        trans   = np.random.choice(transmissions,   p=trans_weights)
        drive   = np.random.choice(drive_types,     p=drive_weights)
        year    = np.random.randint(2015, 2025)

        disp_range = {
            'compact': (1.0, 2.5), 'midsize': (1.5, 3.5),
            'fullsize': (2.0, 5.0), 'suv': (1.5, 4.0),
            'truck': (2.5, 6.2), 'van': (2.5, 4.5), 'sports': (1.5, 6.0)
        }
        lo, hi = disp_range[v_class]
        displacement = round(np.random.uniform(lo, hi), 1)

        if displacement < 1.5:   cylinders = int(np.random.choice([3, 4]))
        elif displacement < 2.5: cylinders = int(np.random.choice([4, 5]))
        elif displacement < 4.0: cylinders = int(np.random.choice([4, 6]))
        else:                    cylinders = int(np.random.choice([6, 8, 12]))

        hp_mean    = 50 + displacement * 60 + cylinders * 5
        horsepower = int(np.clip(np.random.normal(hp_mean, 30), 50, 700))

        weight_base = {
            'compact': 2700, 'midsize': 3300, 'fullsize': 4000,
            'suv': 4200, 'truck': 4800, 'van': 4500, 'sports': 3000
        }
        weight = int(np.clip(np.random.normal(weight_base[v_class], 300), 1500, 8000))

        base_mpg = {
            'compact': 32, 'midsize': 28, 'fullsize': 24,
            'suv': 22, 'truck': 18, 'van': 20, 'sports': 26
        }[v_class]

        mpg = base_mpg
        mpg -= (displacement - 2.0) * 2.5
        mpg -= (horsepower - 150) * 0.025
        mpg -= ((weight - 3000) / 500) * 2.0
        mpg += {'auto': 0, 'manual': 1.5, 'cvt': 2.2, 'dct': 1.8}[trans]
        mpg += {'fwd': 0, 'rwd': -1, 'awd': -3.5, '4wd': -5}[drive]
        mpg *= {'gasoline': 1.0, 'diesel': 1.15, 'hybrid': 1.45, 'electric': 2.8}[fuel]
        mpg += (year - 2015) * 0.4
        mpg += np.random.normal(0, 1.2)
        mpg = max(8, round(mpg, 1))

        data.append({
            'vehicle_class': v_class, 'fuel_type': fuel,
            'transmission': trans, 'drive_type': drive,
            'model_year': year, 'displacement': displacement,
            'cylinders': cylinders, 'horsepower': horsepower,
            'curb_weight': weight, 'mpg_combined': mpg
        })

    return pd.DataFrame(data)


class FuelPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_cols = ['vehicle_class', 'fuel_type', 'transmission', 'drive_type']

    def fit_transform(self, df):
        df = df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def transform(self, df):
        df = df.copy()
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])
        return df


def train_model():
    global MODEL, PREPROCESSOR, FEATURE_COLS, MODEL_METRICS

    print("🤖 Training Gradient Boosting Regressor...")
    df = generate_dataset(10000)

    PREPROCESSOR = FuelPreprocessor()
    df_enc = PREPROCESSOR.fit_transform(df)

    FEATURE_COLS = [
        'vehicle_class', 'fuel_type', 'transmission', 'drive_type',
        'model_year', 'displacement', 'cylinders', 'horsepower', 'curb_weight'
    ]
    X = df_enc[FEATURE_COLS]
    y = df_enc['mpg_combined']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    MODEL = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        min_samples_split=10, min_samples_leaf=4,
        subsample=0.85, random_state=42
    )
    MODEL.fit(X_train, y_train)

    y_pred = MODEL.predict(X_test)
    cv     = cross_val_score(MODEL, X, y, cv=5, scoring='r2')

    MODEL_METRICS = {
        'r2':   round(r2_score(y_test, y_pred), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        'mae':  round(float(mean_absolute_error(y_test, y_pred)), 4),
        'cv_mean': round(float(cv.mean()), 4),
        'cv_std':  round(float(cv.std()), 4),
        'train_samples': 8000,
        'test_samples':  2000,
    }

    print(f"✅ Model ready! R²={MODEL_METRICS['r2']}  RMSE={MODEL_METRICS['rmse']} MPG")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'GradientBoostingRegressor',
        'metrics': MODEL_METRICS
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()

        required = [
            'vehicle_class', 'fuel_type', 'transmission', 'drive_type',
            'model_year', 'displacement', 'cylinders', 'horsepower', 'curb_weight'
        ]
        missing = [k for k in required if k not in body]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        input_df = pd.DataFrame([{
            'vehicle_class': body['vehicle_class'],
            'fuel_type':     body['fuel_type'],
            'transmission':  body['transmission'],
            'drive_type':    body['drive_type'],
            'model_year':    int(body['model_year']),
            'displacement':  float(body['displacement']),
            'cylinders':     int(body['cylinders']),
            'horsepower':    int(body['horsepower']),
            'curb_weight':   int(body['curb_weight']),
        }])

        enc_df  = PREPROCESSOR.transform(input_df)
        X       = enc_df[FEATURE_COLS]
        mpg     = round(float(MODEL.predict(X)[0]), 1)
        mpg     = max(8.0, mpg)

        city_mpg    = round(mpg * 0.82, 1)
        hwy_mpg     = round(mpg * 1.14, 1)
        annual_cost = round((12000 / mpg) * 3.55)
        tank_size   = 26 if body['vehicle_class'] == 'truck' else 14
        range_miles = round(mpg * tank_size)
        co2_gkm     = round(8887 / (mpg * 1.60934), 1) if body['fuel_type'] != 'electric' else 0
        co2_annual  = round((12000 / mpg) * 8.887 / 1000, 2) if body['fuel_type'] != 'electric' else 0
        trees       = round(co2_annual / 0.022) if co2_annual > 0 else 0

        if mpg >= 40:   rating, rating_class = 'EXCELLENT', 'excellent'
        elif mpg >= 30: rating, rating_class = 'GOOD',      'good'
        elif mpg >= 22: rating, rating_class = 'AVERAGE',   'average'
        else:           rating, rating_class = 'POOR',      'poor'

        importances = MODEL.feature_importances_
        feat_imp = sorted(
            [{'name': f, 'importance': round(float(v), 4)}
             for f, v in zip(FEATURE_COLS, importances)],
            key=lambda x: x['importance'], reverse=True
        )

        return jsonify({
            'mpg_combined': mpg,
            'city_mpg':     city_mpg,
            'highway_mpg':  hwy_mpg,
            'annual_fuel_cost': annual_cost,
            'range_miles':  range_miles,
            'co2_gkm':      co2_gkm,
            'co2_annual_tons': co2_annual,
            'trees_to_offset': trees,
            'rating':       rating,
            'rating_class': rating_class,
            'feature_importances': feat_imp,
            'model_metrics': MODEL_METRICS,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    train_model()
    print("🚀 FuelIQ API running at http://localhost:8080")
    app.run(debug=True, port=8080)
