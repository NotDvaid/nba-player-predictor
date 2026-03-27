import os
import math
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("nba_pra_data.csv")

feature_cols = [
    "home",
    "minutes",
    "avg_points_last5",
    "avg_rebounds_last5",
    "avg_assists_last5",
    "opp_avg_points_allowed",
    "opp_avg_rebounds_allowed",
    "opp_avg_assists_allowed"
]

targets = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists"
}

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=200),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

X = df[feature_cols]
results = []
best_models = {}

for target_name, target_col in targets.items():
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_r2 = float("-inf")
    best_model_name = None
    best_model = None
    best_preds = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "target": target_name,
            "model": model_name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = model_name
            best_model = model
            best_preds = preds

    best_models[target_name] = {
        "name": best_model_name,
        "model": best_model
    }

    # Save best model
    joblib.dump(best_model, f"{target_name}_model.pkl")

    # Actual vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_preds, alpha=0.7)
    plt.xlabel(f"Actual {target_name.title()}")
    plt.ylabel(f"Predicted {target_name.title()}")
    plt.title(f"{target_name.title()}: Actual vs Predicted ({best_model_name})")
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.tight_layout()
    plt.savefig(f"plots/{target_name}_actual_vs_predicted.png")
    plt.close()

    # Feature importance / coefficients
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        feat_df.to_csv(f"plots/{target_name}_feature_importance.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(feat_df["feature"], feat_df["importance"])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{target_name.title()} Feature Importance ({best_model_name})")
        plt.tight_layout()
        plt.savefig(f"plots/{target_name}_feature_importance.png")
        plt.close()

    elif hasattr(best_model, "coef_"):
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": best_model.coef_
        }).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

        coef_df.to_csv(f"plots/{target_name}_coefficients.csv", index=False)

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print(results_df)
print("\nBest models:")
for target_name, info in best_models.items():
    print(f"{target_name}: {info['name']}")