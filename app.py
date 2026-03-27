from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog

app = Flask(__name__)
CORS(app)

points_model = joblib.load("points_model.pkl")
rebounds_model = joblib.load("rebounds_model.pkl")
assists_model = joblib.load("assists_model.pkl")

dataset = pd.read_csv("nba_pra_data.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/players", methods=["GET"])
def get_players():
    active_players = players.get_active_players()
    player_names = sorted([player["full_name"] for player in active_players])
    return jsonify(player_names)

@app.route("/teams", methods=["GET"])
def get_teams():
    nba_teams = teams.get_teams()
    team_list = sorted(
        [{"abbreviation": t["abbreviation"], "full_name": t["full_name"]} for t in nba_teams],
        key=lambda x: x["full_name"]
    )
    return jsonify(team_list)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        player_name = data.get("player", "").strip()
        opponent = data.get("opponent", "").strip()
        home = int(data.get("home", 0))
        minutes = float(data.get("minutes", 0))

        if "(" in opponent and ")" in opponent:
            opponent = opponent.split("(")[-1].replace(")", "").strip()

        if not player_name:
            return jsonify({"error": "Player name is required."}), 400
        if not opponent:
            return jsonify({"error": "Opponent is required."}), 400

        matching_players = players.find_players_by_full_name(player_name)
        if not matching_players:
            return jsonify({"error": f"Player '{player_name}' not found."}), 404

        player_id = matching_players[0]["id"]

        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2024-25",
            season_type_all_star="Regular Season"
        )
        df = gamelog.get_data_frames()[0]

        if df.empty or len(df) < 5:
            return jsonify({"error": f"Not enough recent data for {player_name}."}), 404

        last5 = df.head(5)

        avg_points_last5 = last5["PTS"].mean()
        avg_rebounds_last5 = last5["REB"].mean()
        avg_assists_last5 = last5["AST"].mean()

        opp_row = dataset[dataset["opponent"] == opponent]
        if opp_row.empty:
            opp_avg_points_allowed = dataset["opp_avg_points_allowed"].mean()
            opp_avg_rebounds_allowed = dataset["opp_avg_rebounds_allowed"].mean()
            opp_avg_assists_allowed = dataset["opp_avg_assists_allowed"].mean()
        else:
            opp_avg_points_allowed = opp_row["opp_avg_points_allowed"].mean()
            opp_avg_rebounds_allowed = opp_row["opp_avg_rebounds_allowed"].mean()
            opp_avg_assists_allowed = opp_row["opp_avg_assists_allowed"].mean()

        features = pd.DataFrame([{
            "home": home,
            "minutes": minutes,
            "avg_points_last5": avg_points_last5,
            "avg_rebounds_last5": avg_rebounds_last5,
            "avg_assists_last5": avg_assists_last5,
            "opp_avg_points_allowed": opp_avg_points_allowed,
            "opp_avg_rebounds_allowed": opp_avg_rebounds_allowed,
            "opp_avg_assists_allowed": opp_avg_assists_allowed
        }])

        predicted_points = points_model.predict(features)[0]
        predicted_rebounds = rebounds_model.predict(features)[0]
        predicted_assists = assists_model.predict(features)[0]
        predicted_pra = predicted_points + predicted_rebounds + predicted_assists

        return jsonify({
            "predicted_points": round(predicted_points, 1),
            "predicted_rebounds": round(predicted_rebounds, 1),
            "predicted_assists": round(predicted_assists, 1),
            "predicted_pra": round(predicted_pra, 1),
            "avg_points_last5": round(avg_points_last5, 1),
            "avg_rebounds_last5": round(avg_rebounds_last5, 1),
            "avg_assists_last5": round(avg_assists_last5, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)