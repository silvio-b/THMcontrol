import pandas as pd
import os
import pickle

df = pd.read_csv(r'C:\Users\LUCA SANDRI\PycharmProjects\THMcontrol\EplusEnvs\001_EnergyOptimization\Model\FF_genopt_SUBDAILY.csv')
path=r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs'
df.to_csv(os.path.join(path,r'26_simulation.csv'),header=True, encoding = 'utf8', index=False)

rewards=[]
with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\rewards', "rb") as fp:   # Unpickling
    rewards = pickle.load(fp)
with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\26_rewards', "wb") as fp:   #Pickling
    pickle.dump(rewards, fp)

print(rewards[0:10])