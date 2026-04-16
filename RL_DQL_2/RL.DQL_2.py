import os
import sys
import random
import csv
import numpy as np
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CMD = [
    'sumo-gui',
    '-c', 'RL_2.sumocfg',
    '--step-length', '1',
    '--delay', '0',
    '--time-to-teleport', '300',
    '--start'
]

TOTAL_STEPS = 6000
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MIN_GREEN_STEPS = 15
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.0005
YELLOW_DURATION = 4

N2_NS_GREEN = 0
N2_NS_YELLOW = 2
N2_EW_GREEN = 5
N2_EW_YELLOW = 7

N5_MAIN_GREEN = 0
N5_MAIN_YELLOW = 2
N5_SIDE_GREEN = 6
N5_SIDE_YELLOW = 8

DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2", 
    "Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2", 
    "Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2",
    "Node2_5_SB_Det_0", "Node2_5_SB_Det_1", "Node2_5_SB_Det_2",
    "Node4_5_EB_0", "Node4_5_EB_1", "Node4_5_EB_2",
    "Node6_5_WB_0", "Node6_5_WB_1", "Node6_5_WB_2"
]

STATE_SIZE = len(DETECTORS) + 2
ACTION_SIZE = 2 
ACTIONS = [0, 1]

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTIONS)
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        act_values = self.model(state_tensor, training=False)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch], dtype=np.float32)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch], dtype=np.float32)
        next_states = np.array([i[3] for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch])

        q_current = self.model(states, training=False).numpy()
        q_next = self.model(next_states, training=False).numpy()

        indices = np.arange(batch_size)
        q_target = q_current.copy()
        q_target[indices, actions] = rewards + GAMMA * np.max(q_next, axis=1) * (1 - dones)

        self.model.fit(states, q_target, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN: self.epsilon *= EPSILON_DECAY

class TrafficSimulation:
    def __init__(self):
        self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        self.step = 0
        self.last_switch_step = 0
        self.emerg_start_times = {} 
        self.csv_writer = None

    def get_state(self):
        state = []
        for det_id in DETECTORS:
            try: q = traci.lanearea.getLastStepVehicleNumber(det_id)
            except: q = 0
            state.append(q)
        
        ped_count = 0
        try:
            for edge in traci.edge.getIDList():
                if ":Node2" in edge: ped_count += traci.edge.getLastStepPersonNumber(edge)
        except: pass
        state.append(ped_count)

        try: 
            p_raw = traci.trafficlight.getPhase("Node2")
            p = 1 if p_raw >= 5 else 0
        except: p = 0
        state.append(p)
        
        return np.array(state, dtype=np.float32)

    def get_reward(self, state_array):
        q_list = state_array[:len(DETECTORS)]
        cost = np.sum(np.power(q_list, 1.3)) 
        return -cost

    def get_co2_kg(self):
        total_mg = 0
        try:
            for edge in traci.edge.getIDList():
                if not edge.startswith(":"):
                    total_mg += traci.edge.getCO2Emission(edge)
        except: pass
        return total_mg / 1_000_000.0

    def get_avg_wait_time(self):
        total_wt = 0
        veh_count = 0
        for v in traci.vehicle.getIDList():
            total_wt += traci.vehicle.getWaitingTime(v)
            veh_count += 1
        return total_wt / veh_count if veh_count > 0 else 0

    def record_metrics(self):
        try:
            state = self.get_state()
            q_total = np.sum(state[:len(DETECTORS)])
            co2_val = self.get_co2_kg()
            avg_wt = self.get_avg_wait_time()
            
            amb_res, fire_res = "", ""
            for veh_id in traci.simulation.getDepartedIDList():
                v_type = traci.vehicle.getTypeID(veh_id)
                if v_type in ["ambulance", "firetruck"]:
                    self.emerg_start_times[veh_id] = self.step
            
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in self.emerg_start_times:
                    duration = self.step - self.emerg_start_times.pop(veh_id)
                    if "amb" in veh_id or "ambulance" in veh_id:
                        amb_res = duration
                        print(f"🚑 Karetka {veh_id} dotarła! Czas: {duration}s")
                    elif "fire" in veh_id or "firetruck" in veh_id:
                        fire_res = duration
                        print(f"🚒 Straż {veh_id} dotarła! Czas: {duration}s")

            if self.csv_writer:
                self.csv_writer.writerow({
                    'step': self.step, 
                    'total_queue': q_total, 
                    'avg_wait_time': round(avg_wt, 2),
                    'co2_kg': round(co2_val, 3), 
                    'amb_time': amb_res, 
                    'fire_time': fire_res,
                    'reward': 0, 
                    'epsilon': self.agent.epsilon
                })
        except: pass

    def change_phase_with_yellow(self, tls_id, current_phase_idx, target_phase_idx):
        yellow_phase = -1
        if tls_id == "Node2":
            if current_phase_idx == N2_NS_GREEN: yellow_phase = N2_NS_YELLOW
            elif current_phase_idx == N2_EW_GREEN: yellow_phase = N2_EW_YELLOW
            else: yellow_phase = (current_phase_idx + 1) % 10
        elif tls_id == "Node5":
            if current_phase_idx == N5_MAIN_GREEN: yellow_phase = N5_MAIN_YELLOW
            elif current_phase_idx == N5_SIDE_GREEN: yellow_phase = N5_SIDE_YELLOW
            else: yellow_phase = (current_phase_idx + 1) % 10

        try:
            traci.trafficlight.setPhase(tls_id, yellow_phase)
            for _ in range(YELLOW_DURATION):
                traci.simulationStep()
                self.step += 1
                self.record_metrics()
        except: pass

        traci.trafficlight.setPhase(tls_id, target_phase_idx)
        if tls_id == "Node2":
            self.last_switch_step = self.step

    def check_emergency_vehicle(self):
        emergency_veh = [v for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) in ["ambulance", "firetruck"]]
        
        if not emergency_veh: return False

        target_tls = ""
        target_phase = -1
        closest_veh = None
        min_dist = 10000

        for veh in emergency_veh:
            try:
                tls_data = traci.vehicle.getNextTLS(veh)
                if not tls_data: continue
                curr_tls, _, dist, _ = tls_data[0]
                
                if curr_tls in ["Node2", "Node5"] and dist < 300:
                    if dist < min_dist:
                        min_dist = dist
                        closest_veh = veh
                        target_tls = curr_tls
                        edge = traci.vehicle.getRoadID(veh)
                        
                        if target_tls == "Node2":
                            if "SB" in edge or "NB" in edge: target_phase = N2_NS_GREEN
                            else: target_phase = N2_EW_GREEN
                        elif target_tls == "Node5":
                            if "SB" in edge or "NB" in edge: target_phase = N5_SIDE_GREEN
                            else: target_phase = N5_MAIN_GREEN
            except: pass

        if closest_veh:
            curr_p = traci.trafficlight.getPhase(target_tls)
            should_switch = False
            if target_tls == "Node2":
                if curr_p != target_phase and curr_p not in [N2_NS_YELLOW, N2_EW_YELLOW]:
                    should_switch = True
            elif target_tls == "Node5":
                if curr_p != target_phase and curr_p not in [N5_MAIN_YELLOW, N5_SIDE_YELLOW]:
                    should_switch = True

            if should_switch:
                print(f"!!! PRIORYTET ({closest_veh} @ {target_tls}) -> ZMIANA !!!")
                self.change_phase_with_yellow(target_tls, curr_p, target_phase)
            else:
                traci.trafficlight.setPhase(target_tls, target_phase)
                if target_tls == "Node2": self.last_switch_step = self.step
            
            if target_tls == "Node2": return True
        return False

    def run(self):
        print("=== START AI (2 SKRZYŻOWANIA) ===")
        
        with open('wyniki_AI_2_skrzyzowania.csv', 'w', newline='') as f:
            self.csv_writer = csv.DictWriter(f, fieldnames=['step','total_queue','avg_wait_time','co2_kg','amb_time','fire_time','reward','epsilon'])
            self.csv_writer.writeheader()
            
            traci.start(SUMO_CMD)
            traci.gui.setSchema("View #0", "real world")
            traci.trafficlight.setPhase("Node2", N2_NS_GREEN)
            current_state = self.get_state()

            while self.step < TOTAL_STEPS:
                emergency_active = self.check_emergency_vehicle()
                action = 0 
                
                if not emergency_active:
                    q_node2 = np.sum(current_state[:12]) 
                    time_since = self.step - self.last_switch_step
                    curr_ph = traci.trafficlight.getPhase("Node2")

                    if time_since > 80 and q_node2 > 120 and curr_ph in [N2_NS_GREEN, N2_EW_GREEN]:
                        new_ph = N2_EW_GREEN if curr_ph == N2_NS_GREEN else N2_NS_GREEN
                        self.change_phase_with_yellow("Node2", curr_ph, new_ph)
                        action = 1 
                    else:
                        action = self.agent.act(current_state)
                        if action == 1 and time_since > MIN_GREEN_STEPS:
                            if curr_ph in [N2_NS_GREEN, N2_EW_GREEN]:
                                new_ph = N2_EW_GREEN if curr_ph == N2_NS_GREEN else N2_NS_GREEN
                                self.change_phase_with_yellow("Node2", curr_ph, new_ph)

                traci.simulationStep()
                self.step += 1
                
                new_state = self.get_state()
                reward = self.get_reward(new_state)
                
                if not emergency_active:
                    self.agent.remember(current_state, action, reward, new_state, False)
                    self.agent.replay(BATCH_SIZE)
                
                current_state = new_state
                self.record_metrics()
                
                if self.step % 100 == 0:
                    q_total = np.sum(new_state[:len(DETECTORS)])
                    print(f"Step: {self.step}, Q_Net: {q_total}, Eps: {self.agent.epsilon:.2f}")

            traci.close()
            print("Koniec.")

if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run()