#I used this python file for opening the simulation and testing it for one lane alone
import os
import sys
import traci
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
SUMO_CFG_PATH = "C:///Users//adity//Projects_of_Aditya//Research_Paper_Project//sumo_files//bengaluru_junction.sumocfg"
LANE_TO_TEST = '111814614#4_2'
SUMO_BINARY = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
sumo_cmd = [SUMO_BINARY, "-c", SUMO_CFG_PATH]
print("--- Starting SUMO Connection Test ---")
try:
    traci.start(sumo_cmd)
    print(f"Successfully connected to SUMO.")
    print(f"Testing for lane: {LANE_TO_TEST}")
    for step in range(300):
        traci.simulationStep()
        try:
            halting_num = traci.lane.getLastStepHaltingNumber(LANE_TO_TEST)
            if step == 0:
                print(f"\nSUCCESS: SUMO knows about lane '{LANE_TO_TEST}'.")
                print("The problem is likely not in your SUMO files but somewhere in your agent's code.")

        except traci.TraCIException:
            print(f"\nFAILURE: SUMO does NOT know about lane '{LANE_TO_TEST}'.")
            print("This confirms the problem is within your SUMO configuration or network files.")
            break 
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    traci.close()
    print("\n--- Test Finished ---")