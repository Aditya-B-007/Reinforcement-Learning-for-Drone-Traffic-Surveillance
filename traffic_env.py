import numpy as np
import sumolib
import traci
import os
import sys
import random

# Custom space classes to mimic Gymnasium's API
class Discrete:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return random.randint(0, self.n - 1)
class Box:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
class SumoTrafficEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, sumo_cfg_path, tls_id, incoming_lanes, phases, steps_per_action=15, use_gui=True):
        self.sumo_cfg_path = os.path.abspath(sumo_cfg_path)
        self.tls_id = tls_id
        self.incoming_lanes = incoming_lanes
        self.phases = phases
        self.num_phases = len(phases)
        self.steps_per_action = steps_per_action
        self.use_gui = use_gui
        self.sumo_binary = None
        self.is_sumo_running = False
        # Action: Choose the next traffic light phase
        self.action_space = Discrete(self.num_phases)
        # Observation: Queue length on each incoming lane
        self.observation_space = Box(
            low=0, high=np.inf, shape=(len(self.incoming_lanes)+1,), dtype=np.float32
        )
        self.max_episode_steps=5000
        self.current_step=0
    def _start_sumo(self):
        """Starts a SUMO simulation instance."""
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        if self.use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')

        try:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg_path, "--tripinfo-output", "tripinfo.xml"])
            self.is_sumo_running = True
        except traci.TraCIException as e:
            print(f"Error starting SUMO: {e}")
            self.is_sumo_running = False
    def _get_obs(self):
        """
        Get the observation from the environment.
        Observation is the vehicle queue length on each incoming lane.
        """
        try:
            queue_lengths = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in self.incoming_lanes]
            current_phase=[traci.trafficlight.getPhase(self.tls_id)]
            return np.array(queue_lengths+current_phase, dtype=np.float32)
        except traci.TraCIException:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    def _get_reward(self):
        """
        Calculate the reward.
        Reward is the negative of the total waiting time of all vehicles.
        """
        try:
            wait_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in self.incoming_lanes)
            return -wait_time
        except traci.TraCIException:
            return 0
    def reset(self, seed=None, options=None):
        if self.is_sumo_running:
            traci.close()
        self._start_sumo()
        traci.load(["-c", self.sumo_cfg_path, "--tripinfo-output", "tripinfo.xml"])
        # Run a few steps to let some vehicles enter the simulation
        try:
            for _ in range(self.steps_per_action):
                traci.simulationStep()
        except traci.TraCIException:
            self.is_sumo_running = False # Ensure we know sumo is closed
        self.current_step=0
        observation = self._get_obs()
        info = {} # No extra info needed for reset
        return observation, info
    def step(self, action):
        if not self.is_sumo_running:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}

        try:
            phase_to_set = self.phases[action]
            traci.trafficlight.setPhase(self.tls_id, phase_to_set)
            for _ in range(self.steps_per_action):
                traci.simulationStep()
            self.current_step += 1
            observation = self._get_obs()
            reward = self._get_reward()
            active_vehicles = traci.simulation.getMinExpectedNumber()
            terminated = active_vehicles == 0
            truncated = self.current_step>=self.max_episode_steps
            info = {}
            return observation, reward, terminated, truncated, info
        except traci.TraCIException:
            self.is_sumo_running = False
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}

    def close(self):
        """Close the SUMO simulation."""
        if self.is_sumo_running:
            traci.close()
            self.is_sumo_running = False
            print("SUMO environment closed.")