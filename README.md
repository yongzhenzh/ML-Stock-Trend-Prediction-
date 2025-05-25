Stock Trend Prediction Using LSTM and SVM Hybrid Model
Author: Yongzhen “Michael” Zhang
Date: April 21, 2025

Overview
This project implements a hybrid machine learning model that combines a Long Short-Term Memory (LSTM) neural network with a Support Vector Machine (SVM) classifier to predict stock price trends for Apple Inc. (AAPL). It leverages historical stock data and technical indicators to first forecast future prices using LSTM, then classifies the predicted trend direction using SVM.

Technologies Used
Python

yfinance

NumPy, Pandas

TensorFlow (Keras)

Scikit-learn

Matplotlib

Methods and Workflow
1. Data Collection and Preprocessing
Collected daily historical stock data for AAPL (2020–2024) using the yfinance API.

Cleaned the dataset by removing missing and duplicate values.

Engineered technical indicators:

Relative Strength Index (RSI)

10-day Moving Average (MA_10)

One-hot encoded day-of-week features.

Applied MinMaxScaler normalization to all numeric inputs.

2. Sequence Preparation
Used a 60-day rolling window to generate input sequences.

Each sample contains 60 timesteps of 12 features.

Target output: the next day’s normalized closing price.

Binary trend labels (1 = up, 0 = down) were derived from predicted prices.

3. LSTM Model
Architecture:

Bidirectional LSTM (64 units) with return_sequences=True

Bidirectional LSTM (32 units)

Dropout layers (0.3 and 0.2)

Dense (32, activation=tanh) → Dense (1, activation=linear)

Loss: Mean Squared Error (MSE)

Optimizer: Adam

4. SVM Trend Classifier
Trend labels derived from LSTM output.

SVM with RBF kernel and class weight balancing.

Trained on 80% of data, tested on the remaining 20%.

Results
LSTM Regression Performance

MSE: 0.00375

RMSE: 0.0613

SVM Classification Performance

Accuracy: 76%

F1-score:

Up trend: 0.78

Down trend: 0.73

The LSTM effectively captured price behavior, while the SVM reliably classified directional trends.

Limitations and Future Work
A real-time prediction pipeline was not implemented due to time constraints.

Hyperparameter tuning was not performed (e.g., grid search or walk-forward validation).

Potential improvements include:

Adding indicators such as MACD or Bollinger Bands

Trying alternative classifiers like XGBoost or Random Forest

Testing with live data and walk-forward validation

References
Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.

Brownlee, J. (2017). Introduction to Time Series Forecasting with Python. Machine Learning Mastery.

Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques. Expert Systems with Applications, 42(1), 259-268.

Summary
This project demonstrates the value of hybrid modeling in time-series forecasting. LSTM was used to capture long-term dependencies in stock price behavior, and SVM translated the regression outputs into clear directional signals. The combined system achieved both numerical accuracy and robust trend classification, showing strong potential for real-world financial applications.

