#%% Import modules
import sys
sys.path.append('../')
from utils import *

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import dates 

#%%
x = ['1º','2º','3º','4º','5º']
mae = [72.088,103.880, 121.533, 133.607, 142.530]
std = [244.842, 324.385, 359.542, 380.931, 395.264]
#%%
my_dir = os.path.join(".","..","..","db","plots","minute")

if not os.path.exists(my_dir):
    os.makedirs(my_dir)
#%%
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(x, mae,linestyle='-',color= 'red',label = 'MAE', linewidth=2, marker = "o")
ax1.plot(x, std,linestyle='-', color= 'royalblue', label = 'STD', linewidth=2, marker = "s")

ax1.tick_params(axis='x', labelsize= 18)
ax1.tick_params(axis='y', labelsize= 18)

ax1.set_ylabel("Watts", fontsize = 20)
ax1.set_xlabel("Minute", fontsize = 20)

plt.legend(loc='best', fontsize = 'small')
ax1.set_ylim([0, 500])

plt.savefig(my_dir+f'/minute.pdf', format="pdf", bbox_inches="tight", dpi = 600)

#plt.savefig('minute.pdf', format="pdf", bbox_inches="tight", dpi = 600)

# %%