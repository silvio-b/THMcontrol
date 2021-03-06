import pyEp
import os
import time as tm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import cm
from collections import OrderedDict

cmaps = OrderedDict()

def random_slice(df, max_index=8500):         #to check the outouts of manipulated data
    x = np.random.randint(low=1, high=max_index, size=4)
    y = x + 1
    z = x + 2
    random_indexes = x.tolist() + y.tolist() + z.tolist()
    random_indexes.sort()
    return df.iloc[random_indexes]

        #enable to print more columns
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',8)


        #limits of state space
print('__________________________________________________________')
legend = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Legend.csv')
limits = legend[ legend.iloc[:,-1] == legend.iloc[:,-1].min() ]
limits = limits.drop(limits.columns[0], axis=1)
print(limits)
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        #Read and prepare dataset
df = pd.read_csv(r'C:\Users\LUCA SANDRI\PycharmProjects\THMcontrol\EplusEnvs\001_EnergyOptimization\Model\FF_genopt_SUBDAILY.csv')
q_table = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q_table.csv')
q_table_hot = None
print(q_table)
#print('\n'.join(['    '.join(["{:.5f}".format(item) for item in row])
#      for row in q_table]))
#with open(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\rewards', "rb") as fp:   # Unpickling
#    rewards = pickle.load(fp)

df.columns = ['Date/Time', 'Outdoor Temp', 'Diffuse Rad', 'Direct Rad', 'Occupants',
       'EC state', 'nhrs_occ', 'nhrs_sim',
        'UDI_f', 'UDI_s', 'UDI_a', 'UDI_e',                                 #a autonomous --> Only solar radiation,  e excess,   f fell short,   s supplementary --> lighting required  a+s= USEFULL
        'DGPoutputcost', 'DGPoutputcostocc', 'DGPcost', # ???
       'ILLoutput', 'Ep heat', 'Ep cool', 'Ep light', #ILLoutput
       'Termination', 'Warmup',
       'Primary energy', 'PV energy opt',
        'DayOfWeek', 'TimeOfDay', 'Zone Mean Air Temperature']
df = df[ ['Date/Time', 'Outdoor Temp', 'Diffuse Rad', 'Direct Rad', 'Occupants', 'EC state',
          'UDI_s', 'UDI_a', 'Ep heat', 'Ep cool', 'Ep light', 'Primary energy', 'DGPoutputcostocc',
          'DayOfWeek', 'TimeOfDay', 'Zone Mean Air Temperature'] ]
df['EC state'] = df['EC state'].astype(int)
 #df['Reward'] = rewards
 #r_min = df['Reward'].min()
 #r_scaled = df['Reward'] / r_min
 #df['Cumulated R'] = r_scaled.cumsum(axis=0)


        #Combine informations
df['UDI'] = df['UDI_a']*4 + df['UDI_s']*4                       #Get hourly UDI parameters
df['DGP'] = df['DGPoutputcostocc']*4                            #Get DGP parameters
df['Glare'] = np.where(df['DGP']>0.35, 1, 0)                    #Where Glare disconfort occurs
df.drop(labels=['UDI_s', 'UDI_a', 'DGPoutputcostocc'], axis=1, inplace=True)
df['Month'] = 0                                             #Create Month column
df['DayOfMonth'] = 0                                        #Create Day Of the Month column
df['DayOfYear'] = 0                                        #Create Day Of the Month column
for i in df.index:
    df.loc[i, 'DayOfYear'] = df['Date/Time'].iloc[i].split()[0]
    df.loc[i,'Month'] = int(df.loc[i, 'DayOfYear'].split('/')[0])
    df.loc[i, 'DayOfMonth'] = int(df.loc[i, 'DayOfYear'].split('/')[1])
    if not (df.loc[i, 'TimeOfDay'] % 1) == 0:
        prev = df.loc[(i-1), 'TimeOfDay']
        next = df.loc[(i+1), 'TimeOfDay']
        df.loc[i, 'TimeOfDay'] = (prev+next)/2

        #PRINT: Performance parameters
EPh = df['Ep heat'].sum()
EPc = df['Ep cool'].sum()
EPl = df['Ep light'].sum()
Ep = df['Primary energy'].sum()

df_occ = df.where(df.Occupants > 0.0).dropna()              #Working hours only
df_empty = df.where(df.Occupants == 0).dropna()

df_cold = df[7296:].append(df[:2520], ignore_index=True)    #Cold season only
df_hot = df[2520:7296].reset_index().drop(columns='index')  #Hot season only

UDI = df_occ['UDI'].mean()*100
DGP = df_occ['DGP'].mean()*100
Glares = df_occ['Glare'].mean()*100

Results = [EPh, EPc, EPl, Ep, UDI, DGP, Glares]
RBC_Results = [20.55, 1.16, 7.35, 29.06, 54.83, 33.734, 26.901]
vs_RBC = []
for i in range(len(Results)):
    improvement = round( (RBC_Results[i] - Results[i])*100 / RBC_Results[i] , 2 )
    vs_RBC.append((str(improvement) + ' %'))
    Results[i] = round(Results[i], 2)
UDI_improv = round( - (RBC_Results[4] - Results[4])*100 / RBC_Results[4] , 2 )
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
Performance.to_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Performance.csv')

inputs = pd.read_csv(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Inputs.csv')
print(inputs)

        #Organize: by Day
def byDay(df):
    df_byDay = df.drop(['Date/Time', 'DayOfWeek', 'TimeOfDay'], axis=1)
    mode = ['mean']*5 + ['sum']*4 + ['mean']*2 + ['sum'] + ['mean']*2 + ['sum']
    aggregation = dict(zip(df_byDay.columns, mode))
    df_byDay = df_byDay.groupby(['Month','DayOfMonth']).agg(aggregation)
    return df_byDay

df_occ_daily = byDay(df_occ)
df_empty_daily = byDay(df_empty)
df_hot_daily = byDay(df_hot)
df_cold_daily = byDay(df_cold)
df_daily = byDay(df)

def byWeek(df):
    if not len(df) == 365:
        print('Error: Organizing by week requires a daily index')
    else:
        df_byWeek = df.reset_index().drop(['Month','DayOfMonth'], axis=1)
        mode = ['mean']*5 + ['sum']*4 + ['mean']*2 + ['sum'] + ['mean']*2 + ['sum']
        aggregation = dict(zip(df_byWeek.columns, mode))
        df_byWeek = df_byWeek.groupby(df_byWeek.index // 7).agg(aggregation)
        return df_byWeek

#df_weekly = byWeek(df_daily)

#_________________________________________________________________________________

#           Ricompensa e tutti EP per settimana
plt.close()
#fig1, ax1 = plt.subplots()
#weeks = list(df_weekly.index)
#p1 = plt.bar(weeks, df_weekly['Ep heat'])
#p2 = plt.bar(weeks, df_weekly['Ep cool'], bottom=df_weekly['Ep heat'])
#p3 = plt.bar(weeks, df_weekly['Ep light'], bottom=df_weekly['Ep cool'])
#reward_curve = df_weekly['Reward'] * -100
#learning = plt.plot(weeks, reward_curve,  linewidth=4, color='red')
#plt.ylabel('Ep by service')
#plt.legend((p1[0], p2[0], p3[0]), ('Ep heat', 'Ep cool', 'Ep light'))

df['TimeOfDay'] = df['TimeOfDay'].astype('int')
carpet = df.pivot(index='DayOfYear', columns='TimeOfDay', values='EC state')
#carpet_hot = df_hot.pivot(index='DayOfYear', columns='TimeOfDay', values='EC state')
#carpet_cold = df_cold.pivot(index='DayOfYear', columns='TimeOfDay', values='EC state')
fig2, ax = plt.subplots()
sns.heatmap(carpet, ax=ax, xticklabels=4)
ax.set_title('Actions selected through the year')
fig2.set_figwidth(3)
fig2.savefig(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Carpet plot',
             bbox_inches='tight',
             dpi=250)
#sns.heatmap(carpet_hot, ax=ax_hot)
#ax_hot.set_title('Hot season')
#sns.heatmap(carpet_cold, ax=ax_cold)
#ax_cold.set_title('Cold season')

fig3, q_ax = plt.subplots()
sns.heatmap(q_table, ax=q_ax)
q_ax.set_title('Q-table', fontsize=20)
if not q_table_hot:
    q_ax.set_title('Q-table', fontsize=20)
else:
    q_ax.set_title('Q-table for cold season', fontsize=40)
#sns.heatmap(q_table_hot, ax=q_ax_hot)
#q_ax_hot.set_title('Q-table for hot season', fontsize=40)
q_ax.tick_params(labelsize=10)
fig3.set_figwidth(3)
fig3.savefig(r'C:\Users\LUCA SANDRI\Desktop\Tesi\000_PRATICA\Outputs\Current\Q-table plot',
             bbox_inches='tight',
             dpi=250)

#q_ax_hot.tick_params(labelsize=20)


#           Ricompensa e EP(h) per settimana
#fig2, ax2 = plt.subplots()
#weeks = list(df_weekly.index)
#p1 = plt.bar(weeks, df_weekly['Ep heat'])
#reward_curve = df_weekly['Reward'] * -100
#learning = plt.plot(weeks, reward_curve,  linewidth=4, color='red')
#plt.ylabel('Ep by service')
#plt.legend((p1[0], p2[0], p3[0]), ('Ep heat', 'Ep cool', 'Ep light'))


def Weekly_vs_EP_Reward(df):
    fig, ax = plt.subplots()
    weeks = list(df.index)
    p1 = plt.bar(weeks, df['Ep heat'])
    p2 = plt.bar(weeks, df['Ep cool'], bottom=df['Ep heat'])
    p3 = plt.bar(weeks, df['Ep light'], bottom=df['Ep cool'])
    #reward_curve = df['Reward'] * -100
    #learning = plt.plot(weeks, reward_curve, linewidth=4, color='red')
    plt.ylabel('Ep by service')
    plt.legend((p1[0], p2[0], p3[0]), ('Ep heat', 'Ep cool', 'Ep light'))
    return fig, ax

#fig3, ax3 = plt.subplots()
#x_axis = list(range(len(df_cold_daily.index)))
#reward_curve = df_cold_daily['Reward'] * -100
#plt.plot(x_axis, reward_curve,  linewidth=1, color='red')
#
#fig4, ax4 = plt.subplots()
#x_axis = list(range(len(df_hot_daily.index)))
#reward_curve = df_hot_daily['Reward'] * -100
#plt.plot(x_axis, reward_curve,  linewidth=1, color='red')

#f, (ax3, ax4, ax5) = plt.subplots(1, 3, sharey=True)
#x_axis = list(range(len(df_daily.index)))
#reward_curve = df_daily['Reward']
#ax3.plot(x_axis, reward_curve,  linewidth=1, color='red')
#ax3.set_title('All the year')

#x_axis = list(range(len(df_cold_daily.index)))
#reward_curve = df_cold_daily['Reward']
#ax4.plot(x_axis, reward_curve,  linewidth=1, color='red')
#ax4.set_title('Cold season')
#x_axis = list(range(len(df_hot_daily.index)))
#reward_curve = df_hot_daily['Reward']
#ax5.plot(x_axis, reward_curve,  linewidth=1, color='red')
#ax5.set_title('Hot season')



        #PLOT: Curve of Primary energy, UDI - by Day NOT WORKING
#impacts = df_daily[['EC state','Primary energy','UDI']]
#impacts.loc[:,'UDI'] = impacts.UDI * 100
#impacts.loc[:,'EC state'] = impacts.loc[:,'EC state'] * 25
#print(impacts.head())
#impacts.plot.line()

#     #Graph: Ep VS Direct radiation
#fig1, ax1 = plt.subplots()
#EpVSDirect = ax1.scatter(df['Direct Rad'], df['Primary energy']*1000)
#ax1.set_xlabel("Direct solar Radiation [W/m2]")
#ax1.set_xscale('log')
#ax1.set_xlim([0.1,1100])
#ax1.set_ylabel("Primary energy [Wh/m2]")
#ax1.set_yscale('log')
#ax1.set_ylim([0.001,100])

#        #Graph: EP(l) VS Outdoor Temperature
#fig2, ax2 = plt.subplots()
#EPlVSOutdoor = ax2.scatter(df['Outdoor Temp'], df['Ep light'])
#ax2.set_xlabel("Outdoor Temp [°C]")
#ax2.set_ylabel("Ep light [kWh/m2]")

#        #Graph: EP(l) VS Outdoor Temperature
#fig3, ax3 = plt.subplots()
#ECstateVSEp = ax3.scatter(df['EC state'], df['Primary energy']*1000) #Useless, need time line
#ax3.set_xlabel("EC state [1=darkest]")
#ax3.set_ylabel("Primary energy [Wh/m2]")
#ax3.set_yscale('log')
#ax3.set_ylim([0.001,100])



#        #PLOT: Colour map of Actions - by Day NOT WORKING
#state_fig, state_ax = plt.subplots()
#state_fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#state_ax.imshow(df_daily['EC state'], aspect='auto', cmap=plt.get_cmap('Greys'))


#state_fig, state_ax = plt.subplots()
#state_fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#state_ax.imshow(df_byDay['EC state'], aspect='auto', cmap=plt.get_cmap('Greys'))
#
## Plotting lines
#plt.plot(x1, y1, label = "line 1")
## line 2 points
#x2 = [10,20,30]
#y2 = [40,10,30]
## plotting the line 2 points
#plt.plot(x2, y2, label = "line 2")
#plt.xlabel('x - axis')
## Set the y axis label of the current axis.
#plt.ylabel('y - axis')
## Set a title of the current axes.
#plt.title('Two or more lines on same plot with suitable legends ')
## show a legend on the plot
#plt.legend()
## Display a figure.
#plt.show()


plt.show()








'''All_UDIs = pd.unique(df['UDI_s'])
All_UDIa = pd.unique(df['UDI_a'])
#print( "UDI_s can be: ", All_UDIs, " and UDI_a can be: ", All_UDIa)

All_DGPout = pd.unique(df['DGPoutputcost'])
All_DGPout = np.sort(All_DGPout)
All_DGPocc = pd.unique(df['DGPoutputcostocc'])
All_DGPocc = np.sort(All_DGPocc)
All_DGPcost = pd.unique(df['DGPcost'])
#print( "DGPoutputcost can be: ", All_DGPout, " and DGPoutputcostocc can be: ", All_DGPocc, " and DGPcost can be: ", All_DGPcost)

Direct_tot = df['Direct Rad'].sum()
Diffuse_tot = df['Diffuse Rad'].sum()
#print("Ratio between Diffuse/Direct is: ")'''

#     #PERCENTILI DI RADIAZIONE
#day = df[df['TimeOfDay'].between(9,17)]
#DirRad25, DirRad50, DirRad75 = day['Direct Rad'].quantile([.33, .5, .75])
#print("During daytime, DirRad25 = {:.5}, DirRad50 = {:.5}, DirRad75 = {:.5}".format( DirRad25, DirRad50, DirRad75))