import pyEp
import os
import time as tm
import numpy as np
import pandas as pd
import os
from Agents.TabQ import TabularQLearning


selection_type = 0      # 0:'greedy', 1:'softmax', 2:'desc softmax'(not working yet)
seasons = 1             # 0: no, 1: yes
occupancy = 0          # 0: no, 1: yes, 2: Rule Based
n_I_states = 4
I_levels = 1            # 0: equally distributed, 1: fixed limits (100-250-700)
n_zoneT_states = 2
n_extT_states = 0

# TabQ Learning Parameters
learning_rate = 0.99                                       # Structural parameter
gamma = 0.99                                            # Structural parameter
n_actions = 4                                           # Structural parameter
eps = 0.1
temperature = 1
season = 0

if seasons == 0:
    def get_season():
        season = 0
        return season
    def get_zoneT_level(zoneT):
        if zoneT < 20:
            zoneT_level = 0
        else:
            zoneT_level = 1
        return zoneT_level
else:
    first_step_hot = 24 * (31 + 28 + 31 + 30) + 1  # prima ora di Maggio
    last_step_hot = first_step_hot + 24 * (31 + 30 + 31 + 31 + 30)  # prima ora di Ottobre
    def get_season():
        season = TQL.season
        if kStep == first_step_hot:
            print('Season flag to 1')
            season = 1
        if kStep == last_step_hot:
            print('Season flag to 0')
            season = 0
        return season
    def get_zoneT_level(zoneT):
        if TQL.season == 0:
            if zoneT < 18:
                zoneT_level = 0
            else:
                zoneT_level = 1
        elif TQL.season == 1:
            if zoneT < 20:
                zoneT_level = 0
            else:
                zoneT_level = 1
        return zoneT_level

if selection_type == 1:
    def get_action(next_state_index):
        action = TQL.get_softmax_action(next_state_index)
        return action
elif selection_type == 0:
    def get_action(next_state_index):
        action = TQL.get_greedy_action(next_state_index)
        return action
elif selection_type == 2:
    def get_action(next_state_index):
        action = TQL.get_softmax_action(next_state_index)
        return action

if occupancy == 0 or occupancy == 1:
    def action_selection(next_state_index):
        BCVTB_THM_CONTROL = get_action(next_state_index)
        return BCVTB_THM_CONTROL
else:
    def action_selection(next_state_index):
        if OCC == 1:  # Se Vuoto: rule-based
            BCVTB_THM_CONTROL = get_action(next_state_index)
        else:
            if TQL.season == 0:
                BCVTB_THM_CONTROL = 3
            else:
                BCVTB_THM_CONTROL = 0
        return BCVTB_THM_CONTROL

if I_levels == 0:
    def get_I_level(I_tot):
        return int(np.ceil(I_tot/(1000/(n_I_states-1))))
else:
    def get_I_level(I_tot):
        if I_tot < 250:
            if I_tot < 100:
                I_level = 0
            else:
                I_level = 1
        else:
            if I_tot < 700:
                I_level = 2
            else:
                I_level = 3
        return I_level

I_states = [x for x in range(0, n_I_states, 1)]
zoneT_states = [x for x in range(0, n_zoneT_states, 1)]
OCC_states = [0, 1]
states = []
if not occupancy == 0:
    if not n_zoneT_states == 0:
        for i in OCC_states:
            for j in zoneT_states:
                for z in I_states:
                    states.append((i, j, z))
        print('States are by Occupacy, Temperature and Radiation:')
        print(states)
        def get_state(OCC,zoneT_level, I_level):
            return (OCC,zoneT_level, I_level)
    else:
        for i in OCC_states:
            for z in I_states:
                states.append((i, z))
        print('States are by Occupacy and Radiation:')
        print(states)
        def get_state(OCC,zoneT_level, I_level):
            return (OCC, I_level)
else:
    if not n_zoneT_states == 0:
        for j in zoneT_states:
            for z in I_states:
                states.append((j, z))
        print('States are by Temperature and Radiation:')
        print(states)
        def get_state(OCC,zoneT_level, I_level):
            return (zoneT_level, I_level)
    else:
        for z in I_states:
            states.append((z))
        print('States are by Radiation:')
        print(states)
        def get_state(OCC,zoneT_level, I_level):
            return I_level

n_states = len(states)

TQL = TabularQLearning(lr=learning_rate,
                       gamma=gamma,
                       eps=eps,
                       temperature=temperature,
                       action_space_shape=n_actions,
                       state_space_shape=n_states,
                       season=season)
TQL.season = 0

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
I_Dir = output[3]
I_Diff = output[4]
I_tot = I_Dir + I_Diff
I_prev = I_tot
zoneT = output[2]

I_level = get_I_level(I_tot)
zoneT_level = get_zoneT_level(zoneT)
OCC = 0 if output[6] == 0 else 1
state = get_state(OCC,zoneT_level, I_level)
state_index = states.index(state)

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

    output = ep.decode_packet_simple(ep.read()) # columns = ['Time','Day','OutdoorTemp', 'DirectRad', 'DiffuseRad', 'ZoneTemp', 'Occupancy',
                                                #           'EPh', 'EPc', 'EPl', 'ECwindow']
    TIME = output[0]
    DAY = output[1]
    I_Dir = output[3]
    I_Diff = output[4]
    I_tot = I_Dir + I_Diff
    zoneT = output[2]

    I_level = get_I_level(I_prev)
    I_prev = I_tot
    #I_level = get_I_level(I_tot)
    zoneT_level = get_zoneT_level(zoneT)
    OCC = 0 if output[6] == 0 else 1
    next_state = get_state(OCC,zoneT_level, I_level)

    TQL.season = get_season()

    next_state_index = states.index(next_state)

    reward = - output[7] - output[8] - output[9]
    rewards.append(reward)

    TQL.update_table(state_index=state_index,
                     action_index=BCVTB_THM_CONTROL,
                     next_state_index=next_state_index,
                     reward=reward)

    BCVTB_THM_CONTROL = action_selection(next_state_index)

    inputs = [BCVTB_THM_CONTROL + 1]
    input_packet = ep.encode_packet_simple(inputs, time)
    ep.write(input_packet)

    state_index = next_state_index

    kStep = kStep + 1

tm.sleep(10)
ep.close()

import pickle
with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\rewards', "wb") as fp:   #Pickling
    pickle.dump(rewards, fp)

np.save(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\cold_table.npy', TQL.q_table_cold)
np.save(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\hot_table.npy', TQL.q_table_hot)

print('Q table of Cold season:')
print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
      for row in TQL.q_table_cold]))
print('Q table of Hot season:')
print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
      for row in TQL.q_table_hot]))