from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import pandas as pd
import time

season = "2024-25"
max_players = 60
all_rows = []

active_players = players.get_active_players()[:max_players]

for player in active_players:
    player_name = player["full_name"]
    player_id = player["id"]

    try:
        print(f"Pulling data for player ID {player_id}...")

        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )

        df = gamelog.get_data_frames()[0]

        if df.empty or len(df) < 6:
            print(f"Skipping player ID {player_id}: not enough data")
            continue

        df = df.copy()
        df["HOME"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
        df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
        df = df.dropna(subset=["MIN", "PTS", "REB", "AST"])

        if len(df) < 6:
            print(f"Skipping player ID {player_id}: not enough usable rows")
            continue

        for i in range(5, len(df)):
            previous_5 = df.iloc[i-5:i]
            matchup = df.iloc[i]["MATCHUP"]
            opponent = matchup.split()[-1]

            all_rows.append({
                "player_name": player_name,
                "opponent": opponent,
                "home": int(df.iloc[i]["HOME"]),
                "minutes": float(df.iloc[i]["MIN"]),
                "avg_points_last5": previous_5["PTS"].mean(),
                "avg_rebounds_last5": previous_5["REB"].mean(),
                "avg_assists_last5": previous_5["AST"].mean(),
                "points": float(df.iloc[i]["PTS"]),
                "rebounds": float(df.iloc[i]["REB"]),
                "assists": float(df.iloc[i]["AST"])
            })

        time.sleep(0.8)

    except Exception as e:
        print(f"Error with player ID {player_id}: {e}")
        time.sleep(1)

dataset = pd.DataFrame(all_rows)

opp_allowed = (
    dataset.groupby("opponent")[["points", "rebounds", "assists"]]
    .mean()
    .reset_index()
    .rename(columns={
        "points": "opp_avg_points_allowed",
        "rebounds": "opp_avg_rebounds_allowed",
        "assists": "opp_avg_assists_allowed"
    })
)

dataset = dataset.merge(opp_allowed, on="opponent", how="left")
dataset.to_csv("nba_pra_data.csv", index=False)

print("\nDataset saved as nba_pra_data.csv")
print(dataset.head())
print(f"Total rows: {len(dataset)}")
print(f"Total players used: {dataset['player_name'].nunique() if not dataset.empty else 0}")