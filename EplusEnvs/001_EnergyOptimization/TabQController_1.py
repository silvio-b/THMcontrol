import pyEp
import os
import time as tm
import numpy as np
import pandas as pd
import os
from Agents.TabQ import TabularQLearning

pyEp.set_eplus_dir("C:\\EnergyPlusV9-2-0")

directory = os.path.dirname(os.path.realpath(__file__))

path_to_buildings = os.path.join(directory)
# 'C:\\Users\\SilvioBrandi\\Desktop'
builder = pyEp.socket_builder(path_to_buildings)
configs = builder.build()  # Configs is [port, building_folder_path, idf]
weather_file = 'GBR_London.Gatwick.037760_IWEC'
ep = pyEp.ep_process('localhost', configs[0][0], configs[0][1], weather_file)

#	Cosimulation
outputs = []

EPTimeStep = 1
SimDays = 365
kStep = 0
MAXSTEPS = int(SimDays * 24 * EPTimeStep) + 1
deltaT = (60 / EPTimeStep) * 60

# TabQ Learning Parameters
learning_rate = 1
gamma = 0.99
n_actions = 4
n_states = 6 # 0, 0-200, 200-400, 400-600, 600-800, 800-1000

TQL = TabularQLearning(lr=learning_rate,
                       gamma=gamma,
                       action_space_shape=n_actions,
                       state_space_shape=n_states)

print("Running Cosimulation with Total Steps " + str(MAXSTEPS))

# Process first output
output = ep.decode_packet_simple(ep.read())
I_Dir = output[3]
I_Diff = output[4]
state = I_Dir + I_Diff

state_index = int(np.ceil(state/200))
BCVTB_THM_CONTROL = 3
inputs = [BCVTB_THM_CONTROL + 1]
input_packet = ep.encode_packet_simple(inputs, 0)
ep.write(input_packet)
kStep = kStep + 1


while kStep < MAXSTEPS:
    time = (kStep - 1) * deltaT
    dayTime = time % 86400
    if dayTime == 0:
        print(kStep)

    output = ep.decode_packet_simple(ep.read())

    TIME = output[0]
    DAY = output[1]
    I_Dir = output[3]
    I_Diff = output[4]
    next_state = I_Dir + I_Diff

    next_state_index = int(np.ceil(next_state / 200))

    reward = - output[7] - output[8] - output[9]

    TQL.update_table(state_index=state_index,
                     action_index=BCVTB_THM_CONTROL,
                     next_state_index=next_state_index,
                     reward=reward)

    BCVTB_THM_CONTROL = TQL.get_greedy_action(next_state_index)
    inputs = [BCVTB_THM_CONTROL + 1]
    input_packet = ep.encode_packet_simple(inputs, time)
    ep.write(input_packet)

    state_index = next_state_index

    kStep = kStep + 1

tm.sleep(20)
ep.close()

print(TQL.q_table)