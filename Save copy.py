import pandas as pd
import os
import pickle
import numpy as np
from PIL import Image


path=r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Exploiting'
df = pd.read_csv(r'C:\Users\LUCA SANDRI\PycharmProjects\THMcontrol\EplusEnvs\001_EnergyOptimization\Model\FF_genopt_SUBDAILY.csv')
q_table = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q_table.csv')
legend = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Legend.csv')
inputs = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Inputs.csv')
Performance = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Performance.csv')
Carpet_year = Image.open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Carpet plot.png')
Carpet_Q = Image.open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q-table plot.png')

q_table = q_table.round(3)

df.to_csv(os.path.join(path,r'2.11_simulation.csv'),header=True, encoding = 'utf8', index=False)
q_table.to_csv(os.path.join(path,r'2.11_Q_Table.csv'),header=True, encoding = 'utf8', index=False)
legend.to_csv(os.path.join(path,r'2.11_legend.csv'),header=True, encoding = 'utf8', index=False)
inputs.to_csv(os.path.join(path,r'2.11_inputs.csv'),header=True, encoding = 'utf8', index=False)
Performance.to_csv(os.path.join(path,r'2.11_Performance.csv'),header=True, encoding = 'utf8', index=False)
Carpet_year.save(os.path.join(path,r'2.11_Carpet plot.png'))
Carpet_year.save(os.path.join(path,r'2.11_Q-table plot.png'))

#rewards=[]
#with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\rewards', "rb") as fp:   # Unpickling
#    rewards = pickle.load(fp)
#with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Exploiting\2.11_rewards', "wb") as fp:   #Pickling
#    pickle.dump(rewards, fp)