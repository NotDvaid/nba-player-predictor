async function loadPlayers() {
  try {
    const response = await fetch("http://127.0.0.1:5000/players");
    const players = await response.json();

    const playerList = document.getElementById("players");
    playerList.innerHTML = "";

    players.forEach(player => {
      const option = document.createElement("option");
      option.value = player;
      playerList.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading players:", error);
  }
}

async function loadTeams() {
  try {
    const response = await fetch("http://127.0.0.1:5000/teams");
    const teams = await response.json();

    const teamList = document.getElementById("teams");
    teamList.innerHTML = "";

    teams.forEach(team => {
      const option = document.createElement("option");
      option.value = `${team.full_name} (${team.abbreviation})`;
      teamList.appendChild(option);
    });
  } catch (error) {
    console.error("Error loading teams:", error);
  }
}

async function predict() {
  const player = document.getElementById("player").value;
  const opponentRaw = document.getElementById("opponent").value;
  const location = document.getElementById("location").value;
  const minutes = document.getElementById("minutes").value;

  const match = opponentRaw.match(/\(([A-Z]+)\)$/);
  const opponent = match ? match[1] : opponentRaw;

  const home = location === "home" ? 1 : 0;

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        player: player,
        opponent: opponent,
        home: home,
        minutes: minutes
      })
    });

    const data = await response.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    document.getElementById("points").innerText = data.predicted_points;
    document.getElementById("rebounds").innerText = data.predicted_rebounds;
    document.getElementById("assists").innerText = data.predicted_assists;
    document.getElementById("pra").innerText = data.predicted_pra;
  } catch (error) {
    console.error("Prediction error:", error);
  }
}

window.onload = function () {
  loadPlayers();
  loadTeams();
};