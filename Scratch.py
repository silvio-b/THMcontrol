import numpy as np
import pandas as pd

q_table_cold = np.load(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\cold_table.npy')
q_table_hot = np.load(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\hot_table.npy')
states = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]



print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
      for row in TQL.q_table_cold]))
print(q_table_cold)