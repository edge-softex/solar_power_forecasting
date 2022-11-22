#%%
from utils import mae_multi, root_mean_square_error, standard_deviation_error
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
                    choices={0, 1},
                    default=1,
                    help ='Neural network topology which the dataset will be prepared for (0. MLP or 1. LSTM).')
  
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
else:
    network = 'lstm'
layers_list = args.layers_list
input_labels = args.input_labels
output_labels = args.output_labels
n_steps_in = args.input_steps
n_steps_out = args.output_steps
version = 1

#%%
#Testing the model fully trained
#Reading the test input data
input_test = pd.read_csv(r'./../db/data/testInputData.csv')
#Getting the column values
input_test = input_test.values
#Reading the training output data
output_test = pd.read_csv(r'./../db/data/testOutputData.csv')
output_test = output_test.values

if network == 'lstm':
    in_l = len(input_labels)
    out_l = len(output_labels)
    input_test = input_test.reshape(input_test.shape[0], int(input_test.shape[1]/in_l), in_l)

#%%
#my_dir = os.path.join(".","..","db","models",f"{network}", '_'.join(str(e) for e in layers_list), f"{version}")
my_dir = os.path.join(".","..","db","models",f"{network}", "kt_best_model","model.h5")


model = tf.keras.models.load_model(my_dir)

predictions = model.predict(input_test)

normalizator = joblib.load(r'./../db/norm/normPotencia_FV_Avg.save')

#%%
shift = 850

y = normalizator.inverse_transform(output_test[0+shift:1440+shift])
y_hat = normalizator.inverse_transform(predictions[0+shift:1440+shift])    

y_true = np.array(y)
y_pred = np.array(y_hat)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 20})

t = np.arange(0, len(y_true), 1)
times=np.array([datetime.datetime(2022, 1, 3, int(p/60), int(p%60), int(0)) for p in t])
fmtr = dates.DateFormatter("%H:%M")

plt.style.use('seaborn-whitegrid')

plot_size = y_true.shape[1]
#%%
for i in range(plot_size):
    minute = i + 1
    fig = plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(1, 1, 1)
    ax1.plot(times,y_true[:,i],linestyle='-',color= 'red',label = 'Real', linewidth=1.5)
    ax1.plot(times,y_pred[:,i],linestyle='--', color= 'royalblue', label = 'Predito', linewidth=2.5,dashes=(1, 2))
    ax1.xaxis.set_major_formatter(fmtr)
    
    ax1.tick_params(axis='x', labelsize= 18)
    ax1.tick_params(axis='y', labelsize= 18)
    
    ax1.set_ylabel("Potência (W)", fontsize = 20)
    ax1.set_xlabel("Hora", fontsize = 20)
    #plt.title("Gráfico Real x Predito - Minuto "+str(minute), fontsize = 18)
    plt.legend(fontsize = 'small', loc='upper right')
    #plt.grid(b=True)
    #plt.savefig(save_path, dpi=300)
    plt.savefig(f'previsao_{i+1}.pdf', format="pdf", bbox_inches="tight", dpi = 600)

# %%
