import pyEp
import os
import time as tm
import numpy as np
import pandas as pd
import os

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
charge_flag = 0
bill = []

grad_last = 0.0000000001
grad_act = 0.0000000001
RAD_last = 0
RAD_act = 0

while kStep < MAXSTEPS:
    time = (kStep - 1) * deltaT
    dayTime = time % 86400
    if dayTime == 0:
        print(kStep)

    output = ep.decode_packet_simple(ep.read())
    #columns = ['Time','Day','OutdoorTemp', 'DirectRad', 'DiffuseRad', 'ZoneTemp', 'Occupancy',
    #           'EPh', 'EPc', 'EPl', 'ECwindow']

    TIME = output[0]
    DAY = output[1]
    OCC = output[6]
    RAD_dir = output[3]

    if RAD_dir < 250:
        if RAD_dir < 100:
            BCVTB_THM_CONTROL = 4
        else:
            BCVTB_THM_CONTROL = 3
    else:
        if RAD_dir < 700:
            BCVTB_THM_CONTROL = 2
        else:
            BCVTB_THM_CONTROL = 1

    inputs = [BCVTB_THM_CONTROL]
    input_packet = ep.encode_packet_simple(inputs, time)
    ep.write(input_packet)

    kStep = kStep + 1

tm.sleep(20)
ep.close()


#EPh_tot = table.head(50).loc('EPh').sum()
#print("Total EP for heating: %10.3 and Total EP for lighting: " % (table.head(50)['EPh']))


