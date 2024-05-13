import os
from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load models and scalers for each stock symbol
models = {}
feature_scalers = {}
target_scalers = {}

symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "CVS"]
models_dir = "models"

for symbol in symbols:
    models[symbol] = load_model(os.path.join(models_dir, f"{symbol}_lstm_model.h5"))
    feature_scalers[symbol] = joblib.load(os.path.join(models_dir, f"{symbol}_feature_scaler.pkl"))
    target_scalers[symbol] = joblib.load(os.path.join(models_dir, f"{symbol}_target_scaler.pkl"))

@app.route('/')
def home():
    return render_template('stock_index.html', symbols=symbols)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form.get("symbol")
    feature_names = ["open_price", "close_price", "low_price", "high_price", "volume"]

    if symbol not in models:
        return render_template('stock_index.html', symbols=symbols, prediction_text="Invalid stock symbol.")

    # Extract and scale features
    raw_features = [float(request.form.get(name, 0)) for name in feature_names]
    scaled_features = feature_scalers[symbol].transform([raw_features]).reshape(1, 1, -1)

    # Predict with the appropriate model
    prediction = models[symbol].predict(scaled_features)
    predicted_value = round(target_scalers[symbol].inverse_transform(prediction)[0][0], 2)

    return render_template('stock_index.html', symbols=symbols, prediction_text=f'Predicted stock price for {symbol} is {predicted_value}')

if __name__ == "__main__":
    app.run(debug=True)
