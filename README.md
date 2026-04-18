# 🏀 NBA Player Performance Predictor

A machine learning application that predicts NBA player performance (Points, Rebounds, Assists, and PRA) using historical game data and real-time inputs.

---

## Overview

This project leverages machine learning models to forecast NBA player performance for upcoming games. Predictions are based on recent player trends, game context, and opponent strength.

The system integrates live NBA data with trained models and delivers predictions through a web-based interface.

---

## Features

- Predict Points, Rebounds, Assists, and PRA  
- Uses rolling averages from the last 5 games  
- Incorporates opponent defensive metrics  
- Real-time player and team lookup via NBA API  
- REST API for predictions  
- Interactive web interface for inputs and results  

---

## Model Pipeline

- Data collection using NBA API  
- Feature engineering:
  - Rolling averages (last 5 games)
  - Player consistency metrics (standard deviation)
  - Efficiency metrics (points per minute)
  - Opponent-adjusted features  
- Model training using:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting  
- Model evaluation:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score  
- Best-performing models saved using `joblib`

---

## Tech Stack

- Python (Flask API)
- scikit-learn
- pandas / numpy
- nba_api
- joblib

---

## How It Works

1. User selects a player and opponent  
2. System retrieves recent game data via NBA API  
3. Features are generated:
   - Last 5 game averages  
   - Player efficiency  
   - Opponent defensive impact  
4. Models generate predictions:
   - Points  
   - Rebounds  
   - Assists  
5. PRA (Points + Rebounds + Assists) is calculated and returned  

---

## Example Output

```json
{
  "points": 24.3,
  "rebounds": 7.8,
  "assists": 5.6,
  "pra": 37.7
}
