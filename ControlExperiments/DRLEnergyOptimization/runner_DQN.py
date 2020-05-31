from ControlExperiments.DRLEnergyOptimization.GYMEnvironments.SimpleEnv1 import THMControlEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
import os
import pandas as pd
import tensorflow as tf

config = {'res_dir':'C:\\Users\\SilvioBrandi\\OneDrive - Politecnico di Torino\\PhD_Silvio\\14_Projects\\003_GlazingControl\\SimulationResults\\prova1',
          'weather_file':'GBR_London.Gatwick.037760_IWEC',
          'idf_dir':'London_Simple1'}

env = DummyVecEnv([lambda: THMControlEnv(config)])  # The algorithms require a vectorized environment to run

policy_kwargs = dict(layers=[128, 128, 128], act_fun=tf.nn.relu)

model = DQN(MlpPolicy, env, gamma=0.99, learning_rate=0.01, buffer_size=24*365*2,
            exploration_fraction=0.8, exploration_final_eps=0.01, train_freq=1, batch_size=32, learning_starts=365,
            target_network_update_freq=1e-3, prioritized_replay=True,
            prioritized_replay_alpha=0.2, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-06, param_noise=False, verbose=0, tensorboard_log=None, _init_setup_model=True,
            policy_kwargs=policy_kwargs, full_tensorboard_log=False)

model.learn(total_timesteps=24*365*50)