#%%
import sys
sys.path.append("..") 

from utils import read_dat_file, get_list_of_files, split_sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
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
# %%
input_labels = ["Radiacao_Avg", "Temp_Cel_Avg", "Temp_Amb_Avg",  "Potencia_FV_Avg", "Demanda_Avg", "FP_FV_Avg", "Tensao_Rede_Avg"]
df_label = df_complete[input_labels]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)

#%%
df_label.columns = ["IRRADIANCE", "PV MODULE TEMP.", "AMBIENT TEMP.", "PV POWER", "DEMAND", "POWER FACTOR", "VOLTAGE"]
corr = df_label.corr()
#mask = np.zeros_like(corr)
#mask[np.triu_indices_from(mask)] = True
matrix = np.triu(np.ones_like(corr))
#%%
my_dir = os.path.join(".","..","db","plots","others")
if not os.path.exists(my_dir):
    os.makedirs(my_dir)
#%%
plt.figure()
sns.set_theme(style='white', font_scale=0.7)
sns.heatmap(corr, annot=True,mask=matrix, fmt='.2f', cmap=sns.diverging_palette(0,255, as_cmap=True))
#plt.title("Correlation between variables",fontsize=12)
plt.savefig(my_dir+f'/correlation.pdf', format="pdf", bbox_inches="tight", dpi = 600)
# %%
