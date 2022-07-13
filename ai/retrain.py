#%%
from utils import mae_multi, root_mean_square_error, standard_deviation_error, init_gpus
import pandas as pd
import argparse, json, datetime, os
import tensorflow as tf

init_gpus()
#%%
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Model Training')
  
# Adding Argument
parser.add_argument('--network',
                    type = int,
                    choices={0, 1},
                    default="0",
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
#%%
#TEMPORARIO
input_training = pd.read_csv(r'./../db/data/trainingInputData.csv')
input_training = input_training.values

output_training = pd.read_csv(r'./../db/data/trainingOutputData.csv')
output_training = output_training.values

if network == 'lstm':
    in_l = len(input_labels)
    out_l = len(output_labels)
    input_training = input_training.reshape(input_training.shape[0], int(input_training.shape[1]/in_l), in_l)
    output_training = output_training.reshape(output_training.shape[0], int(output_training.shape[1]/out_l), out_l)

#%%
save_path = ""

for i in range(len(layers_list)):
    save_path = save_path + "["+str(layers_list[i])+"]"
save_path = save_path +"["+str(n_steps_in) + "]"+"["+str(n_steps_out) + "]"
for i in range(len(input_labels)):
    save_path = save_path + "["+str(input_labels[i]) + "]" 

# Openning the file which contains the network model
nn_file = open(f'./../db/saves/{network}/regressor_{network}'+save_path+'.json', 'r')
nn_structure = nn_file.read()
nn_file.close()
# Getting the network structure
model = tf.keras.models.model_from_json(nn_structure)
# Reading the weights the putting them  in the network model
model.load_weights(f'./../db/saves/{network}/pesos_{network}'+save_path+'.h5')
#%%
# Compilling the network according to the loss_metric
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer = opt, loss = 'mean_absolute_error', metrics=[mae_multi, standard_deviation_error, root_mean_square_error])  
es = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 10, verbose = 1)

# Reduce the learnning rate when the metric stop improving.
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, verbose = 1)

my_dir = os.path.join(".","..","db","saves",f"{network}")
check_folder = os.path.isdir(my_dir)

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(check_folder)

mcp =  tf.keras.callbacks.ModelCheckpoint(filepath=my_dir+f'/pesos_{network}'+save_path+'.h5', monitor = 'val_loss', save_best_only= True)

log_dir = f"logs/{network}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#training and storing the history
history = model.fit(x = input_training,
                        y= output_training,
                        validation_split=0.2, 
                        epochs = 50,
                        batch_size = 512,
                        callbacks = [es,rlr,mcp,tb_callback])
model_json = model.to_json()

        
hist = {'loss': str(history.history['loss']),
        'mae': str(history.history['mae_multi']),
        'rmse': str(history.history['root_mean_square_error']),
        'stddev': str(history.history['standard_deviation_error'])
        }



j_hist = json.dumps(hist)

my_dir = os.path.join(".","..","db","saves",f"{network}")
check_folder = os.path.isdir(my_dir)

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(check_folder)

with open(my_dir+f'/history_{network}'+save_path+'_retrain_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'w') as json_file:
    json_file.write(j_hist)
with open(my_dir+f'/regressor_{network}'+save_path+'.json', 'w') as json_file:
    json_file.write(model_json)

print("training finished!")
# %%
