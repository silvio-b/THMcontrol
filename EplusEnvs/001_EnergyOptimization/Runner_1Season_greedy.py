import pyEp
import time as tm
import numpy as np
import os
from Agents.TabQ import TabularQLearning
from Agents.Spaces import DiscreteSpace


selection_type = 0      # 0:'greedy', 1:'softmax', 2:'desc softmax'(not working yet)
occupancy = 0          # 0: no, 1: yes, 2: Rule Based
n_I_states = 4
I_levels = 1            # 0: equally distributed, 1: fixed limits (100-250-700)
n_zoneT_states = 2
n_extT_states = 0

state_breakpoints = [[-10, 0, 200, 400, 600, 800, 1000],
                     [-10, 0, 5, 10, 15, 20, 25, 30, 40]]

StateSpace = DiscreteSpace(breakpoints=state_breakpoints)

n_actions = 4
n_states = StateSpace.space_dim

# TabQ Learning Parameters
learning_rate = 1                                       # Structural parameter
gamma = 0.5                                            # Structural parameter
eps = 0.1

TQL = TabularQLearning(lr=learning_rate,
                       gamma=gamma,
                       action_space_shape=n_actions,
                       state_space_shape=n_states)

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

print("Running Cosimulation with Total Steps " + str(MAXSTEPS))

# Process first output
output = ep.decode_packet_simple(ep.read())
TIME, DAY, TE, I_DIR, I_DIFF, TI, OCC, EPH, EPC, EPL, ECW = output
I_TOT = I_DIR + I_DIFF
OCC = np.clip(OCC, 0, 1)
state = [I_TOT, TE]

state_index = StateSpace.get_index(values=state)

BCVTB_THM_CONTROL = 3
inputs = [BCVTB_THM_CONTROL + 1]
input_packet = ep.encode_packet_simple(inputs, 0)
ep.write(input_packet)
kStep = kStep + 1

rewards = []

while kStep < MAXSTEPS:
    time = (kStep - 1) * deltaT
    dayTime = time % 86400
    if dayTime == 0:
        print(kStep)

    output = ep.decode_packet_simple(ep.read())
    TIME, DAY, TE, I_DIR, I_DIFF, TI, OCC, EPH, EPC, EPL, ECW = output
    I_TOT = I_DIR + I_DIFF
    OCC = np.clip(OCC, 0, 1)
    next_state = [I_TOT, TE]

    next_state_index = StateSpace.get_index(values=next_state)

    reward = - EPH - EPC - EPL
    rewards.append(reward)

    if kStep < 4400:
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

tm.sleep(10)
ep.close()

print(TQL.q_table)

# import pickle
# with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\rewards', "wb") as fp:   #Pickling
#     pickle.dump(rewards, fp)
#
# np.save(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\cold_table.npy', TQL.q_table_cold)
# np.save(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\hot_table.npy', TQL.q_table_hot)
#
# print('Q table of Cold season:')
# print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#       for row in TQL.q_table_cold]))
# print('Q table of Hot season:')
# print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#       for row in TQL.q_table_hot]))