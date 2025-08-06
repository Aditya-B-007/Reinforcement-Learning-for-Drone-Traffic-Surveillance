# Reinforcement-Learning-for-Drone-Traffic-Surveillance
An autonomous traffic control system using Deep Reinforcement Learning (Rainbow DQN) and drone-based vision, simulated in SUMO.

# AeroTraffic-AI: Drone-Based Traffic Management with Deep Reinforcement Learning

AeroTraffic-AI is an intelligent system designed to mitigate urban traffic congestion by using real-time aerial footage from drones to dynamically control traffic signals. This repository contains the core component: a Deep Reinforcement Learning agent trained in the SUMO (Simulation of Urban MObility) environment to find optimal traffic light policies.

## Core Concept

The system is conceptualized in two phases: Simulation and Deployment.

1.  **Simulation (This Repository):** We use the realistic SUMO traffic simulator to create a digital twin of a complex intersection. A Rainbow DQN agent is trained in this environment. The agent's "vision"—the queue lengths and waiting times—simulates the data that would be provided by a real-world drone.
2.  **Deployment (Future Goal):** The trained agent can be deployed in the real world. A drone positioned above an intersection  would capture live video. A computer vision model would process this feed to extract real-time traffic data, which is then fed to our trained agent to control the physical traffic light.

## Features
- **Simulation Environment**: A complex, real-world junction from Bengaluru, India, modeled in **SUMO**.
- **Reinforcement Learning Agent**: A sophisticated **Rainbow DQN** agent with the following components enabled:
  - Double Q-Learning
  - Prioritized Experience Replay
  - Dueling Network Architecture
  - Multi-step Learning
  - Distributional Q-Learning

## Installation & Setup

**1. Prerequisites:**
   - You must have SUMO installed on your system. You can follow the official [SUMO installation guide](https://sumo.dlr.de/docs/Installing.html).
   - After installation, ensure the `SUMO_HOME` environment variable is correctly set up.

**2. Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```

**3. Install Python Dependencies:**
   The project requires Python 3. Install the necessary packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**1. Configure Training:**
   All hyperparameters for the agent (learning rate, discount factor, exploration rate, etc.) and experiment settings can be adjusted in the `basic_config.py` file.

**2. Start Training:**
   To begin training the agent, run the main script from your terminal:
   ```bash
   python main.py
   ```
   - As the agent trains, progress will be logged to the console.
   - Trained model weights will be saved periodically in the `models/` directory.

## Project Structure
```
.
├── main.py                # Main script to configure and run the training loop
├── Agent.py               # Contains the core logic for the Rainbow DQN agent
├── network.py             # Defines the neural network architecture for the Q-function
├── memory.py              # Implements the Prioritized Experience Replay buffer
├── traffic_env.py         # The custom Gym-like wrapper for the SUMO environment
├── basic_config.py        # A centralized file for all hyperparameters and settings
├── sumo_files/              # Directory for all SUMO-related files (.net.xml, .rou.xml, etc.)
├── .gitignore             # Specifies which files Git should ignore
└── README.md              # This documentation file
```
Inspired and credits to the following paper of which I had to heavily refer :- https://arxiv.org/abs/1710.02298
