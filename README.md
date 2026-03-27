🏀 NBA Player Performance Predictor

A machine learning application that predicts NBA player performance (points, rebounds, assists, and PRA) using historical game data and real-time inputs.



##Overview

This project uses machine learning models to forecast player performance for upcoming games based on recent trends, game context, and opponent data.

The system integrates live NBA data with trained models to generate predictions through a web-based interface.


##Features

- Predict points, rebounds, assists, and PRA  
- Uses recent player performance (last 5 games)  
- Incorporates opponent defensive averages  
- Real-time player and team lookup via NBA API  
- REST API for predictions  
- Web interface for user input and results  



##Model Pipeline

- Data collection using NBA API  
- Feature engineering (rolling averages, game context)  
- Model training using:
  - Linear Regression  
  - Random Forest  
  - Gradient Boosting (XGBoost-style approach)  
- Model evaluation using MAE, RMSE, and R²  
- Best-performing models saved with `joblib`  



##Tech Stack

- Python (Flask API)
- scikit-learn
- pandas / numpy
- NBA API
- joblib (model serialization)



##How It Works

1. User selects a player and opponent  
2. System retrieves recent game data using NBA API  
3. Features are generated (last 5 game averages + opponent stats)  
4. Trained models predict:
   - Points  
   - Rebounds  
   - Assists  
5. PRA is calculated and returned  



##Example Output

```json
{
  "predicted_points": 24.3,
  "predicted_rebounds": 7.8,
  "predicted_assists": 5.6,
  "predicted_pra": 37.7
}
