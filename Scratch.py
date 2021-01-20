import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\LUCA SANDRI\PycharmProjects\THMcontrol\EplusEnvs\001_EnergyOptimization\Model\FF_genopt_SUBDAILY.csv')
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

print(df[['Month', 'DayOfMonth', 'DayOfYear']])
print('Values of day of year are: ', df.TimeOfDay.unique())

carpet = df.pivot(index='DayOfYear', columns='TimeOfDay', values='EC state')
print(carpet)


#carpet = pd.DataFrame(index=df.DayOfYear.unique(),columns=df.TimeOfDay.unique())
#print(carpet)
#for day in df.DayOfYear.unique():
#    slice = df[['DayOfYear','EC state']].where(df.DayOfYear == day).dropna()
#
#    print(result)


    #carpet
    #print(day, hour)

