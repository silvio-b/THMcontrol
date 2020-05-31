import os
import time as tm

import gym
import numpy as np
import pandas as pd
import pyEp
from gym import spaces
from Utility.scaling import ScalingMinMax


class THMControlEnv(gym.Env):

    def __init__(self, config):

        self.directory = os.path.dirname(os.path.realpath(__file__))

        self.idf_name = 'FF_genopt_SUBDAILY'
        self.idf_dir = config['idf_dir']

        if 'res_dir' in config:
            self.res_dir = config['res_dir']
        else:
            self.res_dir = ''

        print("File directory: ", self.directory)
        # EnergyPlus weather file
        if "weather_file" in config:
            self.weather_file = config["weather_file"]
        else:
            self.weather_file = '/weather/SPtMasterTable_587017_2012_amy.epw'

        self.eplus_path = 'C:/EnergyPlusV9-2-0/'

        self.epTimeStep = 1

        self.simDays = 365

        # Number of steps per day
        self.DAYSTEPS = int(24 * self.epTimeStep)

        # Total number of steps
        self.MAXSTEPS = int(self.simDays * self.DAYSTEPS)

        # Time difference between each step in seconds
        self.deltaT = (60 / self.epTimeStep) * 60

        # Outputs given by EnergyPlus, defined in variables.cfg
        self.outputs = []

        # Inputs expected by EnergyPlus, defined in variables.cfg
        self.inputs = []

        # Current step of the simulation
        self.kStep = 0

        # Instance of EnergyPlus simulation
        self.ep = None

        # state can be all the inputs required to make a control decision
        # getting all the outputs coming from EnergyPlus for the time being

        self.observation_mins = np.array([1, 1, -5.9, 0, 0, 12.8, 0, 0, 0, 0, 1])
        self.observation_maxs = np.array([24, 7, 31.3, 881, 472, 32.8, 1, 0.03, 0.02, 0.012, 4])

        self.observation_space = spaces.Box(self.observation_mins,
                                            self.observation_maxs, dtype=np.float32)

        # discrete action space for dqn
        self.action_space = spaces.Discrete(4)
        self.actions = [1,2,3,4]
        self.episode_number = 1

    def step(self, action):

        # current time from start of simulation
        time = (self.kStep - 1) * self.deltaT

        # current time from start of day
        dayTime = time % 86400
        # if dayTime == 0:
        #     print("Day: ", int(self.kStep / self.DAYSTEPS) + 1)

        action_phys = self.actions[action]

        self.inputs = [action_phys]
        input_packet = self.ep.encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet)

        # after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read()
        self.outputs = self.ep.decode_packet_simple(output_packet)
        # print("Outputs:", self.outputs)
        if not self.outputs:
            print("Outputs:", self.outputs)
            print("Actions:", action)
            next_state = self.reset()
            return next_state, 0, False, {}

        # modifiy occupancy flag
        if self.outputs[6] > 0:
            self.outputs[6] = 1

        # REWARD CALCULATIONS

        Epglo = self.outputs[7] + self.outputs[8] + self.outputs[9]

        # if self.outputs[3]+self.outputs[4] == 0:
        #     reward = 0
        #     self.outputs[7] = 0
        #     self.outputs[8] = 0
        #     self.outputs[9] = 0
        # else:
        reward = - Epglo*1000

        # END REWARD CALCULATIONS

        # print(next_state)
        next_state = self.outputs

        next_state = ScalingMinMax(next_state, self.observation_mins, self.observation_maxs, np.array([0]),
                                   np.array([1]))

        if max(next_state)>1:
            print(next_state)

        # print(next_state)
        self.kStep += 1

        done = False
        if self.kStep >= self.MAXSTEPS:
            # requires one more step to close the simulation
            input_packet = self.ep.encode_packet_simple(self.inputs, time)
            self.ep.write(input_packet)
            # output is empty in the final step
            # but it is required to read this output for termination
            output_packet = self.ep.read()
            last_output = self.ep.decode_packet_simple(output_packet)
            print("Finished simulation")
            print("Last action: ", action)
            print("Last reward: ", reward)
            done = True
            self.ep.close()
            tm.sleep(10)
            dataep = pd.read_csv(self.directory + '\\SimulationFiles\\' + self.idf_dir + '\\' +self.idf_name + '.csv')
            Epheat = dataep['EMS:Ep heat [KWH/M2](Hourly)'].sum()
            Epcool = dataep['EMS:Ep cool [KWH/M2](Hourly)'].sum()
            Eplight = dataep['EMS:Ep light [KWH/M2](Hourly)'].sum()
            print('Ep heat: ' + str(Epheat))
            print('Ep cool: ' + str(Epcool))
            print('Ep light: ' + str(Eplight))
            print('Ep global: ' + str(Epheat+Epcool+Eplight))

            dataep.to_csv(path_or_buf=self.res_dir + '/' + 'episode_' + str(self.episode_number) + '.csv',
                          sep=';', decimal=',', index=False)
            self.episode_number = self.episode_number + 1
            self.ep = None

        info = {}
        return next_state, reward, done, info

    def reset(self):
        # stop existing energyplus simulation
        if self.ep:
            print("Closing the old simulation and socket.")
            self.ep.close()  # needs testing: check if it stops the simulation
            tm.sleep(5)
            self.ep = None

        # tm.sleep(30)
        # start new simulation
        print("Starting a new simulation..")
        self.kStep = 0
        pyEp.set_eplus_dir("C:\\EnergyPlusV9-2-0")
        path_to_buildings = os.path.join(self.directory, 'SimulationFiles')
        # idf_dir = 'C:/Users/SilvioBrandi/PycharmProjects/gym-rl/eplus/envs/buildings/1ZoneDataCenter/'
        builder = pyEp.socket_builder(path_to_buildings)
        configs = builder.build()
        self.ep = pyEp.ep_process('localhost', configs[0][0], configs[0][1], self.weather_file)

        self.outputs = np.round(self.ep.decode_packet_simple(self.ep.read()), 1).tolist()

        # modifiy occupancy flag
        if self.outputs[6] > 0:
            self.outputs[6] = 1

        Epglo = self.outputs[7] + self.outputs[8] + self.outputs[9]

        next_state = self.outputs

        next_state = ScalingMinMax(next_state, self.observation_mins, self.observation_maxs, np.array([0]),
                                   np.array([1]))

        return next_state

    def render(self, mode='human', close=False):
        pass