import os
import sys
import random
import numpy as np
import time

# Wyłączamy zbędne logi TensorFlow
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

# --- KONFIGURACJA DEMO ---
SUMO_CMD = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--step-length', '1',
    '--delay', '50',
    '--time-to-teleport', '300'
]

STATE_SIZE = 14
ACTION_SIZE = 2 
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 2
PHASE_EW_GREEN = 5
PHASE_EW_YELLOW = 7
YELLOW_DURATION = 3

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    return model

# Prosty Agent tylko do odtwarzania (bez uczenia)
class DemoAgent:
    def __init__(self):
        self.model = build_model()
        self.epsilon = 0.05

    def act(self, state):
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        act_values = self.model(state_tensor, training=False)
        return np.argmax(act_values[0])

class TrafficSimulationDemo:
    def __init__(self):
        self.agent = DemoAgent()
        self.step = 0
        self.last_switch_step = 0

    def get_state(self):
        state = []
        detectors = [
            "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
            "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2", 
            "Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2", 
            "Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"
        ]
        for det_id in detectors:
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
        
        return np.array(state)

    def change_phase_with_yellow(self, current_phase_idx, target_phase_idx):
        print(f"Zmiana świateł: Faza {current_phase_idx} -> ŻÓŁTE -> {target_phase_idx}")
        
        yellow_phase = PHASE_NS_YELLOW if current_phase_idx == PHASE_NS_GREEN else PHASE_EW_YELLOW
        traci.trafficlight.setPhase("Node2", yellow_phase)
        
        for _ in range(YELLOW_DURATION):
            traci.simulationStep()
            self.step += 1
            time.sleep(0.1)
            
        traci.trafficlight.setPhase("Node2", target_phase_idx)
        self.last_switch_step = self.step

    def check_emergency_vehicle(self):
        emergency_veh = [v for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) in ["ambulance", "firetruck"]]
        
        if not emergency_veh: return False

        target_phase = -1
        closest_veh = None
        min_dist = 10000

        for veh in emergency_veh:
            try:
                tls_data = traci.vehicle.getNextTLS(veh)
                if not tls_data: continue
                curr_tls, _, dist, _ = tls_data[0]
                
                if curr_tls == "Node2" and dist < 300:
                    if dist < min_dist:
                        min_dist = dist
                        closest_veh = veh
                        edge = traci.vehicle.getRoadID(veh)
                        if "SB" in edge or "NB" in edge: target_phase = PHASE_NS_GREEN
                        else: target_phase = PHASE_EW_GREEN
            except: pass

        if closest_veh:
            curr_p = traci.trafficlight.getPhase("Node2")
            if curr_p != target_phase:
                if curr_p not in [PHASE_NS_YELLOW, PHASE_EW_YELLOW]: 
                    print(f"PRIORYTET SŁUŻB ({closest_veh})! WYMUSZENIE ZMIANY!")
                    self.change_phase_with_yellow(curr_p, target_phase)
            else:
                traci.trafficlight.setPhase("Node2", target_phase)
                self.last_switch_step = self.step
            return True 
        return False

    def run(self):
        traci.start(SUMO_CMD)
        traci.gui.setSchema("View #0", "real world")
        traci.trafficlight.setPhase("Node2", PHASE_NS_GREEN)
        
        current_state = self.get_state()

        while self.step < 1000:
            emergency_active = self.check_emergency_vehicle()
            
            if not emergency_active:
                time_since = self.step - self.last_switch_step
                curr_ph = traci.trafficlight.getPhase("Node2")
                
                action = self.agent.act(current_state)
                
                if action == 1 and time_since > 15:
                     if curr_ph in [PHASE_NS_GREEN, PHASE_EW_GREEN]:
                        new_ph = PHASE_EW_GREEN if curr_ph == PHASE_NS_GREEN else PHASE_NS_GREEN
                        self.change_phase_with_yellow(curr_ph, new_ph)

            traci.simulationStep()
            self.step += 1
            current_state = self.get_state()

        traci.close()

if __name__ == "__main__":
    sim = TrafficSimulationDemo()
    sim.run()