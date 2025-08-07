# Reinforcement-Learning-for-Drone-Traffic-Surveillance
An autonomous traffic control system using Deep Reinforcement Learning (Rainbow DQN) and drone-based vision, simulated in SUMO.

# AeroTraffic-AI: Drone-Based Traffic Management with Deep Reinforcement Learning

AeroTraffic-AI is an intelligent system designed to mitigate urban traffic congestion by using real-time aerial footage from drones to dynamically control traffic signals. This repository contains the core component: a Deep Reinforcement Learning agent trained in the SUMO (Simulation of Urban MObility) environment to find optimal traffic light policies.

## Core Concept

The system is conceptualized in two phases: Simulation and Deployment.

1.  **Simulation (This Repository):** I used the realistic SUMO traffic simulator to create a digital twin of a complex intersection. Silk Board in Bengaluru faces heavy traffic everyday with 1.5 KM covering distance taking upto 1.5 hours. A Rainbow DQN agent is trained in this environment. The agent's "vision"—the queue lengths and waiting times—simulates the data that would be provided by a real-world drone. This aids in efficient dispersion of traffic
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
   git clone "https://github.com/Aditya-B-007/Reinforcement-Learning-for-Drone-Traffic-Surveillance/tree/main"
   cd Reinforcement-Learning-for-Drone-Traffic-Surveillance
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
Now a simple question "Why this ?" well this is the reason along with the explanation of what the python files do:-
My Project: A Personal Quest to Beat the Silk Board Traffic

If you've ever been stuck in traffic, especially at a place like Bengaluru's Silk Board junction, you understand the frustration that sparked this project. I got so fed up with the endless, inefficient gridlock that I decided to see if I could build a better solution myself. This document is a deep dive into that personal project: an AI agent I designed from the ground up to intelligently manage traffic.

1. The Big Picture: What's the Goal?

At its core, my project tackles the kind of soul-crushing traffic congestion that plagues places like Silk Board. I set out to build an autonomous agent that could intelligently control a traffic light.

Instead of relying on a fixed timer, my agent observes the live traffic flow and decides, in the moment, which lane most deserves a green light. The "brain" behind this is a Rainbow DQN (Deep Q-Network), a modern reinforcement learning model. I used the SUMO traffic simulator as a safe, virtual environment for my agent to learn through trial and error. The ultimate measure of success? Minimizing the total time that cars spend waiting at the intersection.

2. The Environment: My Digital Test-Bed (traffic_env.py)

This file is the crucial link between my AI agent's abstract thoughts and the simulated "physical" world of SUMO. I designed it to function like a standard AI training environment (specifically, like one from OpenAI's Gym), which makes it easy for the agent to plug into.

Here’s how I designed the agent's senses and abilities:

    Observation Space (What My Agent "Sees"): At any given moment, the agent receives a snapshot of the intersection. I programmed this snapshot to include the two most important pieces of information it needs:

        The number of cars waiting in each lane.

        The current state of the traffic light (i.e., which direction is currently green).
        This gives the agent a complete picture of both the traffic demand and its own current action.

    Action Space (What My Agent "Does"): The agent's choices are simple. It can choose a number (an integer), and I mapped each number to a specific traffic light phase. For example, choosing 0 might turn the North-South lights green, while choosing 1 turns the East-West lights green.

    The Reward Function (How My Agent "Learns"): This is the most critical component for successful training. I defined the agent's reward as the negative total waiting time of all cars. By training the agent to maximize this reward, I'm implicitly teaching it to minimize the waiting time. Every second a car is stuck in traffic, the agent's score gets worse. This creates a powerful penalty for causing gridlock and a strong incentive to keep traffic flowing smoothly.

3. The Brain: The Q-Network (network.py)

This file contains the architecture for my agent's neural network. I didn't just use a basic network; I implemented a Dueling Deep Q-Network, which is a more sophisticated and effective design.

Here’s what makes it special:

    What is a Q-Network? In simple terms, its job is to look at the current state of the intersection and predict the long-term value (the "Q-value") of taking each possible action.

    The Dueling Architecture: This was a key design choice. Instead of having one part of the network figure everything out, I split its thinking into two separate streams:

        The Value Stream: This part answers the question, "Overall, how good is it to be in this current situation?"

        The Advantage Stream: This part answers a more specific question: "How much better is taking this one action compared to all the other possible actions?"

    I then combine the outputs of these two streams to get the final Q-values. This separation helps the network learn more efficiently because it can determine the value of a state (e.g., "a traffic jam is bad") without having to learn the specific outcome of every single action from that state.

    Distributional Output: This is another advanced feature I included. Instead of the network outputting a single number for a Q-value (e.g., "the value is 50"), it outputs a full probability distribution (e.g., "there's a 70% chance the value is around 45 and a 30% chance it's around 60"). This gives the agent a much richer, more stable signal to learn from.

4. The Memory: Learning from the Past (memory.py)

An agent can't learn well if it only focuses on what's happening right now. It needs to remember and learn from its past experiences. For this, I implemented a sophisticated memory system called Prioritized Experience Replay (PER).

    Why Replay Memories? Training a neural network on experiences as they happen, one after the other, is unstable. The data is too correlated. By storing tens of thousands of past experiences in a large "replay buffer" and then training on random mini-batches from that buffer, I break these correlations and stabilize the learning process.

    Prioritized Replay: This is where the memory gets clever. Instead of all memories being equal, I designed the system to give higher priority to experiences where the agent was most "surprised." Surprise is measured by how big the error was between the agent's prediction and what actually happened. These high-error moments are the most valuable learning opportunities, so it makes sense to replay them more often.

    The SumTree: To efficiently sample experiences based on their priority, I used a data structure called a SumTree. It's a clever way to ensure that higher-priority memories are picked more frequently without having to slowly search through the entire memory buffer each time.

    N-Step Returns: To help my agent learn faster, I configured it to look beyond the immediate reward. Instead of just considering the next step, it looks at the cumulative reward over the next 3 steps. This helps it connect actions to their longer-term consequences more quickly.

5. The Conductor: Tying It All Together (Agent.py)

This file is the central nervous system of my project. It acts as the conductor, orchestrating the brain, memory, and environment to work together in harmony.

    Initialization: When the agent is first created, I actually build two identical neural networks: an online_net and a target_net. The online_net is the one that is actively learning. The target_net is a periodic copy of the online one, and its weights are frozen for a set number of steps. This provides a stable target for the learning process, preventing the agent from trying to hit a constantly moving target—a key concept from the Double DQN algorithm.

    Choosing an Action: I implemented the classic epsilon-greedy strategy. At the beginning of its training, the agent acts mostly randomly (it explores). As it gains experience, the chance of it acting randomly (the epsilon value) slowly decreases, and it begins to rely more on its own knowledge to pick the best action (it exploits).

    The Learning Step: This is the most complex part, where the core of the Rainbow algorithm comes to life. Every few steps, this happens:

        A mini-batch of prioritized experiences is pulled from memory.

        Using the Double DQN method, my online_net picks the best action for the next state, but the target_net is used to evaluate the value of that action. This helps prevent the agent from becoming overconfident in its predictions.

        The target distribution from the target_net is calculated and used as a "ground truth" for the online_net to learn from.

        I calculate the loss, which is the difference between the agent's prediction and that target.

        Finally, I use backpropagation to update the online_net's weights to minimize this loss, nudging its predictions to be more accurate.

6. The Execution: Running the Show (main.py, basic_config.py)

main.py is the top-level script that starts everything. It initializes the SUMO environment and my agent, then runs the main training loop for a set number of episodes. Inside this loop, it repeatedly asks the agent for an action, applies it to the simulation, and feeds the result (the new state and reward) back to the agent so it can learn.

To make my life easier, I put all the important settings and hyperparameters—like the learning rate, batch size, and epsilon decay rate—into a single configuration file, basic_config.py. This was excellent practice, as it allowed me to experiment with different settings without having to hunt through all my different code files.
