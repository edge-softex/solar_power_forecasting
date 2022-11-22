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
#%%
my_dir = os.path.join(".","..","db","plots","others")
if not os.path.exists(my_dir):
    os.makedirs(my_dir)
#%%

fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
plt.boxplot(df_complete['Potencia_FV_Avg'],meanprops=meanpointprops,
                        medianprops=medianprops,
                        showmeans=False,
                        meanline=False,
                        whiskerprops=whiskerprops,
                        boxprops=boxprops,
                        flierprops=flierprops)
plt.xticks([1], [''])
ax1.set_ylabel("Power (W)", fontsize = 20)
ax1.set_xlabel("Average Photovoltaic Power", fontsize = 20)
ax1.tick_params(axis='x', labelsize= 16)
ax1.tick_params(axis='y', labelsize= 16)


plt.savefig(my_dir+'/boxplot.pdf', format="pdf", bbox_inches="tight", dpi = 600)
# %%