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
import time

#%%
# Initializing Parser
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Model Test')
  
# Adding Argument
parser.add_argument('--network',
                    type = int,
                    choices={"0", "1", "2"},
                    default="0",
                    help ='Neural network topology which the dataset will be prepared for (0. MLP, 1. RNN or 2. LSTM).')
  
parser.add_argument('--layers_list',
                    nargs='+', 
                    default=[60],
                    help ='Number of neurons each hidden layer will have')


parser.add_argument('--dropout_layers',
                    nargs='+', 
                    default=[],
                    help ='Retention probability of each dropout layer the model will have')


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
dropout_layers = args.dropout_layers
input_labels = args.input_labels
output_labels = args.output_labels
n_steps_in = args.input_steps
n_steps_out = args.output_steps

input_scalers = []
for i in input_labels:
    input_scalers.append(joblib.load(f"./../db/norm/norm{i}.save"))
#%%
#Testing the model fully trained
#Reading the test input data
input_test = pd.read_csv(r'./../db/data/testInputData.csv')
#Getting the column values
input_test = input_test.values
#Reading the training output data
output_test = pd.read_csv(r'./../db/data/testOutputData.csv')
output_test = output_test.values

if network == 'lstm' or network == 'rnn':
    in_l = len(input_labels)
    out_l = len(output_labels)
    input_test = input_test.reshape(input_test.shape[0], int(input_test.shape[1]/in_l), in_l)
#%%
#my_dir = os.path.join(".","..","db","models",f"{network}", '_'.join(str(e) for e in layers_list), "model.h5")
my_dir = os.path.join(".","..","db","models",f"{network}",f"{n_steps_in}in_{n_steps_out}out" , '_'.join(str(e) for e in layers_list),'Dropout_'+'_'.join(str(e) for e in dropout_layers), "model.h5")
#my_dir = os.path.join(".","..","db","models",f"{network}", "kt_best_model","model.h5")
#my_dir = os.path.join(".","..","db","models",f"{network}",'_'.join(str(e) for e in layers_list), "1")

model = tf.keras.models.load_model(my_dir)

#%%

times = []
for j in range(100):
    start_time = time.time()
    for i in range(100):
        predictions = model.predict(input_test[i:i+1])
    times.append(time.time() - start_time)

print("--- %s seconds ---" % np.mean(times))
# %%
