###
import numpy as np
import argparse
from ControlExperiments.DRLEnergyOptimization.GYMEnvironments.SimpleEnv1 import THMControlEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

import os
import pandas as pd

config = {'res_dir':'C:\\Users\\SilvioBrandi\\OneDrive - Politecnico di Torino\\PhD_Silvio\\14_Projects\\003_GlazingControl\\SimulationResults\\prova1',
          'weather_file':'GBR_London.Gatwick.037760_IWEC',
          'idf_dir':'London_Simple1'}

# Get the environment and extract the number of actions.
env = THMControlEnv(config)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128, kernel_initializer="glorot_normal"))
model.add(Activation('relu'))
model.add(Dense(128, kernel_initializer="glorot_normal"))
model.add(Activation('relu'))
model.add(Dense(128, kernel_initializer="glorot_normal"))
model.add(Activation('relu'))
# model.add(Dense(400, kernel_initializer="glorot_normal"))
# model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=24*365*3, window_length=1)
# policy = EpsGreedyQPolicy_custom(param=[1., .01, 24*31*5., 24*31*95.])
policy = LinearAnnealedPolicy(MaxBoltzmannQPolicy(), attr='eps', value_max=.1, value_min=.0, value_test=.0,
                              nb_steps=24*365*45)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=24*31,
               target_model_update=1e-3, policy=policy, gamma=0.9, batch_size=24*7)
dqn.compile(RMSprop(lr=0.001), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort th
dqn.fit(env, nb_steps=24*365*50, visualize=True, verbose=2, log_interval=24)