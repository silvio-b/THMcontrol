import numpy as np
import pandas as pd
import itertools
from Agents.Spaces import DiscreteSpace

q_table = np.load(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q_table.npy')
print(q_table)
Q = pd.DataFrame(q_table)
print(Q)

########################################################################################################

def TestSoftMax():
    temp=1
    scores = np.array([-0.02409 ,   -0.00719 ,   -0.01370   , -0.01938])
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    e_scores = np.exp(norm_scores/temp)
    probs = e_scores / e_scores.sum()
    print(probs)

def TestBreakpoints():
    Rad_breakpoints = [-10, 0, 100, 200, 300, 500, 1000] *1
    ExtTemp_breakpoints = [-10, 10, 15, 20, 26, 30, 40] *1
    Occ_breakpoints = [0,1] *1
    RadFore_breakpoints = [-10,1e3,5e3,1e4,1e6] *0
    Szn_breakpoints = [-1, 1872, 6144, 9000] *0                   #equinozi: 20 Marzo, 23 Settmebre

    all_breakpoints = [Rad_breakpoints,
                       ExtTemp_breakpoints,
                       Occ_breakpoints,
                       RadFore_breakpoints]

    state_breakpoints = []
    for i in all_breakpoints:
        if not len(i) == 0:
            state_breakpoints.append(i)

    original_breakpoints = [[-10, 0, 100, 200, 300, 500, 1000],
                         [-10, 10, 15, 20, 26, 30, 40]]

    print(original_breakpoints==state_breakpoints)

##################################################################################

# TabQ Learning Parameters
learning_rate = 1                                       # Structural parameter
gamma = 0.5                                            # Structural parameter
eps_0 = 0.1
eps = eps_0
eps_cold = eps_0
eps_hot = eps_0
decay = 0.1

Szn_breakpoints = [-1, 1872, 6144, 9000] *1                   #equinozi: 20 Marzo, 23 Settmebre
Rad_breakpoints = [-10, 0, 100, 200, 300, 500, 1000] *1
ExtTemp_breakpoints = [-10, 10, 15, 20, 26, 30, 40] *1
Occ_breakpoints = [-0.5,0.5,1.5] *1
RadFore_breakpoints = [-10,1e3,5e3,1e4,1e6] *1

all_breakpoints = [Rad_breakpoints,
                   ExtTemp_breakpoints,
                   Occ_breakpoints,
                   RadFore_breakpoints]

def TabellaInputs():
    Selezione = 'e={}; d={}'.format(eps_0,decay)
    Stagioni = list((np.array(Szn_breakpoints, dtype=int) / 24).astype('int'))
    print(len(RadFore_breakpoints))
    if len(RadFore_breakpoints) == 0:
        Previsioni = None
    else:
        Previsioni = list(np.array(RadFore_breakpoints, dtype=int).astype('int'))
        Previsioni = str(Previsioni[1:-1]).strip('[').strip(']')
    settings = [gamma, Selezione, str(Stagioni[1:-1]).strip('[').strip(']')]
    for i in all_breakpoints[:-1]:
        if not i:
            settings.append(None)
        else:
            settings.append(str(i[1:-1]).strip('[').strip(']'))
    settings.append(Previsioni)
    print(settings)


    params = pd.Series(['Gamma', 'Selezione','Stagione','Radiazione','Temp esterna','Occupazione','Previsioni'])
    settings = pd.Series(settings)
    inputs = pd.DataFrame({'Parametro': params, 'Input': settings}).dropna()
    print(inputs)

#TabellaInputs()

#############################################################################

def TabellaPerformance():
    Results = [28, 1.2, 4, 30.1, 50, 35, 15]
    RBC_Results = [20.55, 1.16, 7.35, 29.06, 54.83, 33.734, 26.901]
    vs_RBC = []
    for i in range(len(Results)):
        improvement = round( (RBC_Results[i] - Results[i]) / RBC_Results[i] , 2 )
        vs_RBC.append((str(improvement) + ' %'))
    UDI_improv = round( - (RBC_Results[4] - Results[4]) / RBC_Results[4] , 2 )
    vs_RBC[4] = str(UDI_improv) + ' %'

    Performance = ['EP(h) [kWh/m2y]',
                 'EP(c) [kWh/m2y]',
                 'EP(l) [kWh/m2y]',
                 'Ep    [kWh/m2y]',
                 'UDI   [%]',
                 'DGP   [%]',
                 'Glares[%]']
    Performance = pd.DataFrame({'Performance': Performance, 'Test': Results, 'vs RBC': vs_RBC})

    print(Performance)

#TabellaPerformance()
