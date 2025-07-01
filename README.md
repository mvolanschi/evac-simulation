# 🚨 Evacuation Simulation Platform

A web-based simulation tool for modeling evacuation scenarios using heterogeneous agent behaviors—including panic, following, repulsion ignoring, and exit avoidance. Built with a Flask backend and a modern Next.js frontend. Outputs animated MP4 videos and detailed statistics.

![10A395C5-8145-4A5C-A130-5FE50F310E23_1_201_a](https://github.com/user-attachments/assets/a5968fc1-aeb1-4155-a8ef-72f99d2dd50c)
---

## 🔧 Features

- Customizable grid-based environments
- Configurable agent traits and crowd personalities
- Adjustable fire growth, obstacle count, exit layout
- MP4 animation of simulation results
- Downloadable statistics (survival rates, agent breakdown)
- Fully interactive UI (Next.js + TailwindCSS)
- RESTful backend (Flask + Matplotlib)

---

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/mvolanschi/evac-simulation.git
cd evac-simulation
2. Setup Backend (Flask)
cd backend
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Create a .env file (optional):


PORT=5001
Then run the Flask server:


python app.py
The backend will be available at http://localhost:5001.

3. Setup Frontend (Next.js)
cd ../frontend
npm install
Create a .env.local file:
NEXT_PUBLIC_API_BASE=http://localhost:5001
Then start the frontend dev server:
npm run dev
Visit http://localhost:3000 in your browser.

🧪 Simulation Parameters
The frontend UI allows you to configure:

Grid size (rows × columns)

Number of agents (10–50)

Fire growth rate

Agent speed

Number of obstacles

Agent personality distribution:

Follower agents

Repulsion-ignoring agents

Crowded-exit-avoiding agents

Exit positions (up to 4)

Each run generates an MP4 animation and returns statistics including escape rate and fatalities.

📊 Output
MP4 Video: Visual animation of the evacuation

Stats JSON: Includes:

total_agents

escaped, dead

survival_rate, casualty_rate

Agent type breakdown

🧠 Behavioral Model
Agents navigate using:

Potential Fields: Attraction to exits, repulsion from obstacles and nearby agents

Follower Behavior: Blind following of nearby agents

Panic Mode: Introduces noise and speed variation

Exit Avoidance: Avoids crowded exits when configured

Fire Damage: Kills agents caught by expanding fire radius

🤝 Contributions
Feel free to fork and extend the platform. Ideas for future development:

Multi-room or multi-floor environments

ML-based adaptive behaviors

Live heatmap visualization

Real-time crowd control experiments


👤 Author
Developed by G.M. Volanschi
Bachelor Final Project — TU/e (0ISBEP05, 2025–2026)
Supervisor: Dr. K. Cuijpers

