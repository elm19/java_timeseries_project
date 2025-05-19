# Temperature Forecasting System

A Java-based temperature forecasting system that uses machine learning models to predict minimum and maximum temperatures. The system includes both a training component and a graphical user interface for viewing predictions.

## Overview

This project consists of two main components:
1. A machine learning model trainer (`TimeSeriesForecaster.java`)
2. A graphical display interface (`TemperatureDisplay.java`)

The system uses three different machine learning models:
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)

## Features

- Predicts both minimum and maximum temperatures
- Shows 7-day temperature forecast (3 days past, today, and 3 days future)
- Uses multiple ML models and automatically selects the best performing one
- Beautiful, responsive GUI with scrolling support
- Persists trained models for later use
- Cross-validation for model evaluation

## Prerequisites

- Java Development Kit (JDK) 8 or higher
- Required libraries (included in `/lib` folder):
  - Weka (Machine Learning)
  - MTJ (Matrix Operations)
  - Core (Utils)
  - ARPACK (Linear Algebra)
  - Time Series Forecasting

## Project Structure

```
.
├── TimeSeriesForecaster.java   # Model training component
├── TemperatureDisplay.java     # GUI component
├── data/
│   └── daily_temp.csv         # Temperature training data
├── lib/                       # Required libraries
│   ├── arpack_combined.jar
│   ├── core.jar
│   ├── mtj.jar
│   ├── timeseriesForecasting1.0.27.jar
│   └── weka.jar
└── model/                     # Saved trained models
    ├── max/
    │   ├── linear_regression.model
    │   ├── random_forest.model
    │   └── support_vector_regression.model
    └── min/
        ├── linear_regression.model
        ├── random_forest.model
        └── support_vector_regression.model
```

## How to Use

### 1. Training the Models

First, train the models using the temperature data:

```bash
javac -cp "lib/*" TimeSeriesForecaster.java
java -cp "lib/*:." TimeSeriesForecaster
```

This will:
- Load the temperature data from `data/daily_temp.csv`
- Train three different models for both min and max temperatures
- Evaluate models using 10-fold cross-validation
- Save the trained models in the `model/` directory

### 2. Running the Display Interface

After training, run the display interface:

```bash
javac -cp "lib/*" TemperatureDisplay.java
java -cp "lib/*:." TemperatureDisplay
```

The interface shows:
- Current date and relative days (e.g., "Yesterday", "Today", "Tomorrow")
- Min and max temperatures for each day

## Model Evaluation

The system evaluates models using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Relative Absolute Error

The best model is selected based on a weighted combination of these metrics.

## Technical Details

### Data Processing
- Uses time series data with daily minimum and maximum temperatures
- Creates separate models for min and max predictions
- Implements sliding window approach for prediction

### Model Selection
- Weighted scoring system:
  - 40% RMSE
  - 40% MAE
  - 20% Correlation coefficient
- Automatically selects best performing model for predictions

### GUI Features
- Responsive design with minimum size constraints
- Clear visual hierarchy with date and temperature information
- Highlights current day's forecast
