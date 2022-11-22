#%% Import modules
import sys
sys.path.append('../')

from utils import *
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

#%%
folders = [os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2019"), 
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020"),
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2021"),
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2022")]
    

lst = []
for i in range(len(folders)):
    lst = lst + get_list_of_files(folders[i])

dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
#%%
df_power_date = df_complete[['TIMESTAMP', 'Potencia_FV_Avg']]

df_power_date = df_power_date.dropna()
df_power_date = df_power_date.reset_index(drop=True)

indexAux = df_power_date[(df_power_date['Potencia_FV_Avg'] < 0)].index
df_power_date['Potencia_FV_Avg'][indexAux] = 0.0

# %%
df_power_date["TIMESTAMP"] = pd.to_datetime(df_power_date["TIMESTAMP"])
df_power_date["minute"] = df_power_date["TIMESTAMP"].dt.minute
df_power_date["hour"] = df_power_date["TIMESTAMP"].dt.hour
df_power_date["day"] = df_power_date["TIMESTAMP"].dt.day
df_power_date["month"] = df_power_date["TIMESTAMP"].dt.month
df_power_date["time"] = df_power_date["TIMESTAMP"].dt.time
# %%
power_hour = []
power_day = []
power_month = []

for i in range(24):
    power_hour.append(df_power_date[(df_power_date['hour'] == i)]['Potencia_FV_Avg'].values)

for i in range(1,32):
    power_day.append(df_power_date[(df_power_date['day'] == i)]['Potencia_FV_Avg'].values)

for i in range(1,13):
    power_month.append(df_power_date[(df_power_date['month'] == i)]['Potencia_FV_Avg'].values)
#%%
my_dir = os.path.join(".","..","db","plots","others")
if not os.path.exists(my_dir):
    os.makedirs(my_dir)
#%%
plt.style.use('seaborn-whitegrid')
boxprops = dict(linestyle='-',
                linewidth=3,
                color='black')
flierprops = dict(marker='.',
                  markerfacecolor='white',
                  markersize=12,
                  linestyle='none',
                  markeredgecolor='black')
medianprops = dict(linestyle='-',
                   linewidth=4,
                   color='red')
meanpointprops = dict(marker='D',
                      markeredgecolor='green',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='-',
                     linewidth=4, color='green')
whiskerprops = dict(linestyle='-',
                    linewidth=3,
                    color='black')



x = list(range(0,24))

fig = plt.figure(figsize=(16,8))
ax1=fig.add_subplot(1, 1, 1)
plt.boxplot(power_hour, meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
#ax1.set_xticklabels(x, fontsize=13)
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Hour", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

#plt.savefig('power_hour.pdf')
plt.savefig(my_dir+'/distribution_hour.pdf', format="pdf", bbox_inches="tight", dpi = 600)
#plt.grid(b=True)

# %%
x = list(range(1,32))

x = list(range(1,12))

fig = plt.figure(figsize=(16,8))
ax1=fig.add_subplot(1, 1, 1)
plt.boxplot(power_day, meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
#ax1.set_xticklabels(x, fontsize=13)
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Day", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

#plt.savefig('power_day.pdf')
plt.savefig(my_dir+'/distribution_day.pdf', format="pdf", bbox_inches="tight", dpi = 600)

# %%
x = list(range(1,12))

fig = plt.figure(figsize=(16,8))
ax1=fig.add_subplot(1, 1, 1)
plt.boxplot(power_month, meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
#ax1.set_xticklabels(x, fontsize=13)
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Month", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

#plt.savefig('power_month.pdf')
plt.savefig(my_dir+'/distribution_month.pdf', format="pdf", bbox_inches="tight", dpi = 600)


# %%
### Subplot com todos juntos
plt.style.use('seaborn-whitegrid')
boxprops = dict(linestyle='-',
                linewidth=3,
                color='black')
flierprops = dict(marker='.',
                  markerfacecolor='white',
                  markersize=12,
                  linestyle='none',
                  markeredgecolor='black')
medianprops = dict(linestyle='-',
                   linewidth=4,
                   color='red')
meanpointprops = dict(marker='D',
                      markeredgecolor='green',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='-',
                     linewidth=4, color='green')
whiskerprops = dict(linestyle='-',
                    linewidth=3,
                    color='black')

#fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24,22))
fig, (ax1, ax3) = plt.subplots(2, figsize=(24,22))
x = list(range(0,24))


ax1.boxplot(power_hour, meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Hour", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

ax1.set_title('(a)',fontsize=20, fontweight="bold")

x = list(range(1,32))
'''
ax2.boxplot(power_day, meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
ax2.set_ylabel("Potência (W)", fontsize = 20)
ax2.set_xlabel("Dia", fontsize = 20)
ax2.tick_params(axis='x', labelsize= 18)
ax2.tick_params(axis='y', labelsize= 18)

ax2.set_title('(b)',fontsize=20, fontweight="bold")
'''

x = list(range(1,12))

ax3.boxplot(power_month,meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
ax3.set_ylabel("Power (W)", fontsize = 20)
ax3.set_xlabel("Month", fontsize = 20)
ax3.tick_params(axis='x', labelsize= 18)
ax3.tick_params(axis='y', labelsize= 18)


ax3.set_title('(b)',fontsize=20, fontweight="bold")

#plt.savefig('distribuicao.pdf', format="pdf", bbox_inches="tight", dpi = 600)
plt.savefig(my_dir+'/distribution_all.pdf', format="pdf", bbox_inches="tight", dpi = 600)




# %%
index = []
media = []
desv_padr = []

for i in range (24):
    for j in range(60):
        aux = df_power_date[(df_power_date['hour'] == i) & (df_power_date['minute'] == j)]['Potencia_FV_Avg'].values
        media.append(aux.mean())
        desv_padr.append(aux.std())
        index.append(pd.to_datetime(f'{i}:{j}:0', format='%H:%M:%S'))

# %%
zipped = list(zip(index, media, desv_padr))
df = pd.DataFrame(zipped, columns =['Tempo (H:M:S)', 'Média (W)', 'Desvio Padrão (W)'])
df['Tempo (H:M:S)'] = pd.to_datetime(df['Tempo (H:M:S)'])
df['Tempo (H:M:S)'] = [time.time() for time in df['Tempo (H:M:S)']]
print(df)
# %%
os.makedirs('./tabela/', exist_ok=True)  
df.to_csv('./tabela/tabela_horas_minutos.csv')  

# %%
index = []
media = []
desv_padr = []

for i in range (24):
    media.append(power_hour[i].mean())
    desv_padr.append(power_hour[i].std())
    index.append(i)
# %%
zipped = list(zip(index, media, desv_padr))
df = pd.DataFrame(zipped, columns =['Hora', 'Média (W)', 'Desvio Padrão (W)'])
os.makedirs('./tabela/', exist_ok=True)  
df.to_csv('./tabela/tabela_horas.csv')  
# %%
index = []
media = []
desv_padr = []

for i in range (31):
    media.append(power_day[i].mean())
    desv_padr.append(power_day[i].std())
    index.append(i+1)
# %%
zipped = list(zip(index, media, desv_padr))
df = pd.DataFrame(zipped, columns =['Dia', 'Média (W)', 'Desvio Padrão (W)'])
os.makedirs('./tabela/', exist_ok=True)  
df.to_csv('./tabela/tabela_dia.csv')  

# %%
index = []
media = []
desv_padr = []

for i in range (12):
    media.append(power_month[i].mean())
    desv_padr.append(power_month[i].std())
    index.append(i+1)
# %%
zipped = list(zip(index, media, desv_padr))
df = pd.DataFrame(zipped, columns =['Mês', 'Média (W)', 'Desvio Padrão (W)'])
os.makedirs('./tabela/', exist_ok=True)  
df.to_csv('./tabela/tabela_mes.csv')  
# %%