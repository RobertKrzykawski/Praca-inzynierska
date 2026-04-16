import os
import sys
import csv
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CONFIG = [
    'sumo-gui', 
    '-c', 'RL_rondo.sumocfg', 
    '--step-length', '1',
    '--delay', '0',
    '--time-to-teleport', '300',
    '--start'
]

DETECTORS = [
    "det_N_0", "det_N_1",
    "det_E_0", "det_E_1",
    "det_S_0", "det_S_1",
    "det_W_0", "det_W_1"
]

def run_rondo_baseline():
    print("=== START BASELINE RONDO ===")
    
    emerg_start_times = {}
    
    traci.start(SUMO_CONFIG)
    traci.gui.setSchema("View #0", "real world")
    all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]

    with open('wyniki_rondo.csv', 'w', newline='') as csvfile:
        fieldnames = ['step', 'total_queue', 'avg_wait_time', 'co2_kg', 'amb_time', 'fire_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for step in range(1, 6001):
            traci.simulationStep()

            amb_res, fire_res = "", ""

            for veh_id in traci.simulation.getDepartedIDList():
                v_type = traci.vehicle.getTypeID(veh_id)
                if any(x in v_type.lower() or x in veh_id.lower() for x in ["amb", "fire"]):
                    emerg_start_times[veh_id] = (step, v_type)

            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in emerg_start_times:
                    start_step, v_type = emerg_start_times.pop(veh_id)
                    duration = step - start_step
                    
                    if "amb" in v_type.lower() or "amb" in veh_id.lower():
                        amb_res = duration
                        print(f"🚑 Karetka {veh_id} przejechała rondo! Czas: {duration}s")
                    else:
                        fire_res = duration
                        print(f"🚒 Straż {veh_id} przejechała rondo! Czas: {duration}s")

            total_queue = sum(traci.lanearea.getLastStepVehicleNumber(det) for det in DETECTORS)

            all_vehs = traci.vehicle.getIDList()
            veh_count = len(all_vehs)
            total_wt = sum(traci.vehicle.getWaitingTime(v) for v in all_vehs)
            avg_wt = total_wt / veh_count if veh_count > 0 else 0

            total_co2_mg = sum(traci.edge.getCO2Emission(e) for e in all_edges)
            co2_kg = total_co2_mg / 1_000_000.0

            writer.writerow({
                'step': step,
                'total_queue': total_queue,
                'avg_wait_time': round(avg_wt, 2),
                'co2_kg': round(co2_kg, 4),
                'amb_time': amb_res,
                'fire_time': fire_res
            })
            
            if step % 100 == 0:
                print(f"Step: {step} | Queue: {total_queue} | Avg WT: {avg_wt:.2f}s")

        traci.close()

if __name__ == "__main__":
    run_rondo_baseline()