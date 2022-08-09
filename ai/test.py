#%%
from utils import mae_multi, root_mean_square_error, standard_deviation_error
import pandas as pd
import numpy as np
import os
import argparse,joblib
import tensorflow as tf
from keras.models import load_model
#%%
# Initializing Parser
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Model Test')
  
# Adding Argument
parser.add_argument('--network',
                    type = int,
                    choices={0, 1},
                    default=0,
                    help ='Neural network topology which the dataset will be prepared for (0. MLP or 1. LSTM).')
  
parser.add_argument('--layers_list',
                    nargs='+', 
                    default=[60],
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
    output_training = output_test.reshape(output_test.shape[0], int(output_test.shape[1]/out_l), out_l)


# Loading the model
my_dir = os.path.join(".","..","db","models",f"{network}", '_'.join(str(e) for e in layers_list), f"{version}")

model = tf.keras.models.load_model(my_dir)

predictions = model.predict(input_test)

normalizator = joblib.load(r'./../db/saves/norm/normPotencia_FV_Avg.save')

y = normalizator.inverse_transform(output_test)
y_hat = normalizator.inverse_transform(predictions)    

mae = mae_multi(y, y_hat).numpy()
rmse = root_mean_square_error(y, y_hat).numpy()
stddev = standard_deviation_error(y, y_hat).numpy()


print("Tests evaluation:")
for i in range(len(mae)):
    print(str(i+1)+" Output minute:")
    print(f"Mean Absolute Error (MAE): {mae[i]}")
    print(f"Root Mean Square Error (RMS): {rmse[i]}")
    print(f"Standart Deviation: {stddev[i]}")

print("Avarege:")
print(f"Mean Absolute Error (MAE): {np.mean(mae)}")
print(f"Root Mean Square Error (RMS): {np.mean(rmse)}")
print(f"Standart Deviation: {np.mean(stddev)}")
# %%
