import pyEp
import time as tm
import numpy as np
import os
from Agents.TabQ import TabularQLearning
from Agents.Spaces import DiscreteSpace
import pandas as pd


# selection_type = 0      # 0:'greedy', 1:'softmax', 2:'desc softmax'(not working yet)
# occupancy = 0          # 0: no, 1: yes, 2: Rule Based
# n_I_states = 4
# I_levels = 1            # 0: equally distributed, 1: fixed limits (100-250-700)
# n_zoneT_states = 2
# n_extT_states = 0

def get_state(I_TOT, TE, OCC, I_Fore, Szn):
    return [I_TOT, TE]

I_Fore, Szn = 0, 0

Rad_breakpoints = [-10, 100, 200, 300, 500, 1000] *1
ExtTemp_breakpoints = [-10, 10, 20, 26, 40] *1
Occ_breakpoints = [-0.5,0.2,1.5] *0
RadFore_breakpoints = [-10,1e3,5e3,1e4,1e6] *0
Szn_breakpoints = [-1, 1872, 6144, 9000] *0                   #equinozi: 20 Marzo, 23 Settembre
summer = [0,1e6] if len(Szn_breakpoints)==0 else Szn_breakpoints

all_breakpoints = [Rad_breakpoints,
                   ExtTemp_breakpoints,
                   Occ_breakpoints,
                   RadFore_breakpoints]

state_breakpoints = []
for i in all_breakpoints:
    if not len(i) == 0:
        state_breakpoints.append(i)

#state_breakpoints = [[-10, 0, 100, 200, 300, 500, 1000],
#                     [-10, 10, 15, 20, 26, 30, 40]]

StateSpace = DiscreteSpace(breakpoints=state_breakpoints)

n_actions = 4
n_states = StateSpace.space_dim

# TabQ Learning Parameters
learning_rate = 1                                       # Structural parameter
gamma = 0.5                                            # Structural parameter
eps_0 = 0.1
eps = eps_0
eps_cold = eps_0
eps_hot = eps_0
decay = 0.1                         #if 0, eps doesn't change

#if len(Szn_breakpoints) == 0:
#    def get_eps(current, next_eps, step):
#        current = next_eps
#        next_eps = current - decay*current
#        return current, next_eps
#else:
#    def get_eps(current, next_eps, step):
#        if step<Szn_breakpoints[1] or step>Szn_breakpoints[2]:
#            current = next_eps
#            next_eps = current - decay * current
#
#        return current, next_eps


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

state = get_state(I_TOT, TE, OCC, I_Fore, Szn)

state_index = StateSpace.get_index(values=state)

BCVTB_THM_CONTROL = 3
inputs = [BCVTB_THM_CONTROL + 1]
input_packet = ep.encode_packet_simple(inputs, 0)
ep.write(input_packet)
kStep = kStep + 1

Presenze = []

while kStep < MAXSTEPS:
    time = (kStep - 1) * deltaT
    dayTime = time % 86400
    if dayTime == 0:
        print(kStep)

    output = ep.decode_packet_simple(ep.read())
    TIME, DAY, TE, I_DIR, I_DIFF, TI, OCC, EPH, EPC, EPL, ECW = output

    I_TOT = I_DIR + I_DIFF
    OCC = np.clip(OCC, 0, 1)
    next_state = get_state(I_TOT, TE, OCC, I_Fore, Szn)

    next_state_index = StateSpace.get_index(values=next_state)

    reward = - EPH - EPC - EPL
    Presenze.append(OCC)


    TQL.update_table(state_index=state_index,
                     action_index=BCVTB_THM_CONTROL,
                     next_state_index=next_state_index,
                     reward=reward)

    if kStep < summer[1] or kStep > summer[2]:
        eps = eps_cold
        eps_cold = eps - decay * eps
    else:
        eps = eps_hot
        eps_hot = eps - decay * eps
        if dayTime == 0:
            print('HOT')

    BCVTB_THM_CONTROL = TQL.get_e_greedy_action(next_state_index,
                                                eps=eps)

    inputs = [BCVTB_THM_CONTROL + 1]
    input_packet = ep.encode_packet_simple(inputs, time)
    ep.write(input_packet)

    state_index = next_state_index

    kStep = kStep + 1

tm.sleep(10)
ep.close()

#print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#      for row in TQL.q_table]))
legend = StateSpace.get_space_table()
print(legend)

Q_table = pd.DataFrame(TQL.q_table).round(3)
Q_table.to_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q_table.csv',
              header=True, encoding = 'utf8', index=False)
legend.to_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Legend.csv',
              header=True, encoding = 'utf8', index=False)
print(Q_table)


Selezione = 'e={}; d={}'.format(eps_0,decay)
if len(Szn_breakpoints) == 0:
    Stagioni = None
else:
    Stagioni = list((np.array(Szn_breakpoints, dtype=int) / 24).astype('int'))
    Stagioni = str(Stagioni[1:-1]).strip('[').strip(']')
print(len(RadFore_breakpoints))
if len(RadFore_breakpoints) == 0:
    Previsioni = None
else:
    Previsioni = list(np.array(RadFore_breakpoints, dtype=int).astype('int'))
    Previsioni = str(Previsioni[1:-1]).strip('[').strip(']')
settings = [gamma, Selezione, Stagioni]
for i in all_breakpoints[:-1]:
    if not i:
        settings.append(None)
    else:
        settings.append(str(i[1:-1]).strip('[').strip(']'))
settings.append(Previsioni)


params = pd.Series(['Gamma', 'Selezione','Stagione','Radiazione','Temp esterna','Occupazione','Previsioni'])
settings = pd.Series(settings)
inputs = pd.DataFrame({'Parametro': params, 'Input': settings}).dropna()
print(inputs)

inputs.to_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Inputs.csv',
              header=True, encoding = 'utf8', index=False)


#unique = list(set(Presenze))
#unique.sort()
#print(unique)


# np.save(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\hot_table.csv', TQL.q_table_hot)
#
# print('Q table of Cold season:')
# print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#       for row in TQL.q_table_cold]))
# print('Q table of Hot season:')
# print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#       for row in TQL.q_table_hot]))