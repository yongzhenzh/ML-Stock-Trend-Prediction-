# Stock Trend Prediction Using LSTM and SVM Hybrid Model

**Author:** ***Yongzhen “Michael” Zhang***  
**Date:** ***April 21, 2025***

## Overview

This project implements a hybrid machine learning model that combines a ***Long Short-Term Memory (LSTM)*** neural network with a ***Support Vector Machine (SVM)*** classifier to predict stock price trends for ***Apple Inc. (AAPL)***. It leverages historical stock data and technical indicators to first forecast future prices using LSTM, then classifies the predicted trend direction using SVM.

## Technologies Used

- ***Python***
- ***yfinance***
- ***NumPy, Pandas***
- ***TensorFlow (Keras)***
- ***Scikit-learn***
- ***Matplotlib***

## Methods and Workflow

### 1. Data Collection and Preprocessing

- Collected daily historical stock data for ***AAPL (2020–2024)*** using the ***`yfinance`*** API.
- Cleaned the dataset by removing missing and duplicate values.
- Engineered technical indicators:
  - ***Relative Strength Index (RSI)***
  - ***10-day Moving Average (MA_10)***
- One-hot encoded ***day-of-week*** features.
- Applied ***MinMaxScaler*** normalization to all numeric inputs.

### 2. Sequence Preparation

- Used a ***60-day rolling window*** to generate input sequences.
- Each sample contains ***60 timesteps of 12 features***.
- Target output: the ***next day’s normalized closing price***.
- Binary ***trend labels (1 = up, 0 = down)*** were derived from predicted prices.

### 3. LSTM Model

***Architecture:***
- ***Bidirectional LSTM (64 units)*** with `return_sequences=True`
- ***Bidirectional LSTM (32 units)***
- ***Dropout layers (0.3 and 0.2)***
- ***Dense (32, activation=tanh) → Dense (1, activation=linear)***

***Training:***
- ***Loss:*** Mean Squared Error (**MSE**)
- ***Optimizer:*** Adam

### 4. SVM Trend Classifier

- Trend labels derived from ***LSTM output***.
- ***SVM with RBF kernel*** and ***class weight balancing***.
- Trained on ***80%*** of data, tested on the remaining ***20%***.

## Results

### LSTM Regression Performance

- ***MSE:*** 0.00375
- ***RMSE:*** 0.0613

### SVM Classification Performance

- ***Accuracy:*** 76%
- ***F1-score:***
  - ***Up trend:*** 0.78
  - ***Down trend:*** 0.73

The LSTM effectively captured price behavior, while the SVM reliably classified directional trends.

## Source Code

The full implementation is contained in `stock_trend_predictor.py`, which defines and executes the hybrid LSTM-SVM prediction pipeline.

### Main Steps:

**1. Data Acquisition and Preprocessing**
- Fetches historical stock data using `yfinance`.
- Computes technical indicators including the 10-day Moving Average (MA_10) and Relative Strength Index (RSI).
- One-hot encodes weekday features.
- Normalizes features using `MinMaxScaler`.

**2. Sequence Preparation**
- Converts the data into sequences of 60 time steps with 12 input features each.
- The target is the normalized closing price of the next day.

**3. LSTM Model**
- A 3-layer Bidirectional LSTM with dropout is used to model time dependencies.
- The architecture includes:
  - `LSTM(64, return_sequences=True)`
  - `LSTM(32, return_sequences=True)`
  - `LSTM(16, return_sequences=False)`
- Followed by dense layers:
  - `Dense(32, activation='tanh')`
  - `Dense(1, activation='linear')`
- Compiled with `Adam` optimizer and `MSE` loss.

**4. SVM Classifier**
- Uses the predicted LSTM outputs to generate trend labels (1 = up, 0 = down).
- Trains an `SVC` classifier with RBF kernel and balanced class weights.

**5. Evaluation**
- The LSTM model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- The SVM is evaluated using classification accuracy and F1-scores.

**6. Visualization**
- Plots the actual vs. predicted closing prices over the test set.

