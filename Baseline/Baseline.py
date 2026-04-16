import os
import sys
import csv
import traci
import traci.constants as tc

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CONFIG = [
    'sumo-gui', 
    '-c', 'RL.sumocfg',
    '--step-length', '1', 
    '--delay', '0',
    '--time-to-teleport', '300', 
    '--start'
]

DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2", 
    "Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2", 
    "Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"
]

def run_baseline():
    print("=== START BASELINE ===")
    
    start_times = {}
    
    traci.start(SUMO_CONFIG)
    traci.gui.setSchema("View #0", "real world")
    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    with open('wyniki_baseline_single.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'queue', 'avg_wait_time', 'co2_kg', 'ambulance_time', 'firetruck_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(6000):
            traci.simulationStep()
            
            for veh_id in traci.simulation.getDepartedIDList():
                traci.vehicle.subscribe(veh_id, [tc.VAR_TYPE])
                v_type = traci.vehicle.getTypeID(veh_id)
                if v_type in ["ambulance", "firetruck"]:
                    start_times[veh_id] = (step, v_type)

            amb_t, fire_t = "", ""
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in start_times:
                    start_step, v_type = start_times.pop(veh_id)
                    duration = step - start_step
                    
                    if v_type == "ambulance":
                        amb_t = duration
                        print(f"🚑 Karetka {veh_id} przejechała! Czas: {duration}s")
                    else:
                        fire_t = duration
                        print(f"🚒 Straż {veh_id} przejechała! Czas: {duration}s")

            queue = sum(traci.lanearea.getLastStepVehicleNumber(det) for det in DETECTORS)

            all_vehs = traci.vehicle.getIDList()
            veh_count = len(all_vehs)
            total_wt = sum(traci.vehicle.getWaitingTime(v) for v in all_vehs)
            avg_wt = total_wt / veh_count if veh_count > 0 else 0

            total_co2_mg = sum(traci.edge.getCO2Emission(e) for e in all_edges)
            co2_kg = total_co2_mg / 1_000_000.0

            writer.writerow({
                'step': step,
                'queue': queue,
                'avg_wait_time': round(avg_wt, 2),
                'co2_kg': round(co2_kg, 4),
                'ambulance_time': amb_t,
                'firetruck_time': fire_t
            })
            
            if step % 100 == 0:
                print(f"Step: {step} | Queue: {queue} | Avg WT: {avg_wt:.2f}s")

        traci.close()

if __name__ == "__main__":
    run_baseline()