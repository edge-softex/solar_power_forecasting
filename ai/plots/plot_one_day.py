#%%
import sys
sys.path.append("..") 

from utils import *
import pandas as pd
import numpy as np
import os
import argparse,joblib, csv
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import dates 
import datetime
#%%
# Initializing Parser
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Model Test')
  
# Adding Argument
parser.add_argument('--network',
                    type = int,
                    choices={"0", "1", "2"},
                    default="2",
                    help ='Neural network topology which the dataset will be prepared for (0. MLP, 1. RNN or 2. LSTM).')
  
parser.add_argument('--layers_list',
                    nargs='+', 
                    default=[120],
                    help ='Number of neurons each hidden layer will have')


parser.add_argument('--input_labels',
                    nargs='+', 
                    default=["Radiacao_Avg", "Temp_Cel_Avg", "Potencia_FV_Avg"],
                    help ='Input features that will be used to make predictions. (TIMESTAMP, Radiacao_Avg,Temp_Cel_Avg, Temp_Amb_Avg,Tensao_S1_Avg,Corrente_S1_Avg, Potencia_S1_Avg, Tensao_S2_Avg, Corrente_S2_Avg, Potencia_S2_Avg, Potencia_FV_Avg, Demanda_Avg,FP_FV_Avg,Tensao_Rede_Avg')

parser.add_argument('--output_labels',
                    nargs='+', 
                    default=["Potencia_FV_Avg"],
                    help ='Output features to be predicted. (TIMESTAMP, Radiacao_Avg, Temp_Cel_Avg, Temp_Amb_Avg, Tensao_S1_Avg, Corrente_S1_Avg, Potencia_S1_Avg, Tensao_S2_Avg, Corrente_S2_Avg, Potencia_S2_Avg, Potencia_FV_Avg, Demanda_Avg, FP_FV_Avg, Tensao_Rede_Avg')


parser.add_argument('--input_steps',
                    type = int,
                    default=120,
                    help ='Number of minutes used in the input sequence.')

parser.add_argument('--output_steps',
                    type = int,
                    default=5,
                    help ='Number of minutes to be predicted.')
parser.add_argument('-f')

args = parser.parse_args()

if args.network == 0:
    network = 'mlp'
elif args.network == 1:
    network = 'rnn'
else:
    network = 'lstm'


layers_list = args.layers_list
input_labels = args.input_labels
output_labels = args.output_labels
n_steps_in = args.input_steps
n_steps_out = args.output_steps

input_scalers = []
for i in input_labels:
    input_scalers.append(joblib.load(f"./../db/norm/norm{i}.save"))

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
df_complete['TIMESTAMP'] = pd.to_datetime(df_complete['TIMESTAMP'])  
#%%
start_date = '2022-4-18'
end_date = '2022-4-19'
mask = (df_complete['TIMESTAMP'] >= start_date) & (df_complete['TIMESTAMP'] < end_date)
df_day = df_complete.loc[mask]
df_day = df_day[input_labels]
#%%
for i in range(len(input_labels)):
    scaler = input_scalers[i]
    df_day[input_labels[i]] = scaler.transform(df_day[input_labels[i]].values.reshape(-1, 1))

input_test, output_test  = split_sequence(df_day, n_steps_in, n_steps_out,input_labels,output_labels)

#%%
# Loading the model
#my_dir = os.path.join(".","..","db","models",f"{network}", '_'.join(str(e) for e in layers_list), "model.h5")
my_dir = os.path.join(".","..","db","models",f"{network}", "kt_best_model","model.h5")


model = tf.keras.models.load_model(my_dir)

predictions = model.predict(input_test)

#%%
normalizator = joblib.load(r'./../db/norm/normPotencia_FV_Avg.save')

y = normalizator.inverse_transform(output_test.reshape(output_test.shape[0], output_test.shape[1]))
y_hat = normalizator.inverse_transform(predictions)    

y_true = np.array(y)
y_pred = np.array(y_hat)

#%%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 20})

t = np.arange(0, len(y_true), 1)
times=np.array([datetime.datetime(2022, 1, 3, int(p/60), int(p%60), int(0)) for p in t])
fmtr = dates.DateFormatter("%H:%M")

plt.style.use('seaborn-whitegrid')

plot_size = y_true.shape[1]
#%%
my_dir = os.path.join(".","..","db","plots",f"{network}","kt_best_model", f"{start_date}")

if not os.path.exists(my_dir):
    os.makedirs(my_dir)

for i in range(plot_size):
    minute = i + 1
    fig = plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(times,y_true[:,i],linestyle='-',color= 'red',label = 'Real', linewidth=1.5)
    ax1.plot(times,y_pred[:,i],linestyle='--', color= 'royalblue', label = 'Predito', linewidth=2.5,dashes=(1, 2))
    ax1.xaxis.set_major_formatter(fmtr)
    
    ax1.tick_params(axis='x', labelsize= 18)
    ax1.tick_params(axis='y', labelsize= 18)
    
    ax1.set_ylabel("Power (W)", fontsize = 20)
    ax1.set_xlabel("Hour", fontsize = 20)
    #plt.title("Gráfico Real x Predito - Minuto "+str(minute), fontsize = 18)
    plt.legend(fontsize = 'small', loc='upper right')
    #plt.grid(b=True)
    #plt.savefig(save_path, dpi=300)
    
    plt.savefig(my_dir+f'/previsao_{i+1}.pdf', format="pdf", bbox_inches="tight", dpi = 600)
    #plt.savefig(my_dir+f'/previsao_{i+1}.png', bbox_inches="tight", dpi = 600)
# %%
