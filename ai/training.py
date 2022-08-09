#%%
from utils import init_gpus
import pandas as pd
import argparse, datetime, os
import tensorflow as tf

init_gpus()
#%%
# Initializing Parser
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
version = 1

#%%
#Training the model with all the training data
#Reading the training input data
input_training = pd.read_csv(r'./../db/data/trainingInputData.csv')
#Getting the column values
input_training = input_training.values
#Reading the training output data
output_training = pd.read_csv(r'./../db/data/trainingOutputData.csv')
output_training = output_training.values

if network == 'lstm':
    in_l = len(input_labels)
    out_l = len(output_labels)
    input_training = input_training.reshape(input_training.shape[0], int(input_training.shape[1]/in_l), in_l)
    output_training = output_training.reshape(output_training.shape[0], int(output_training.shape[1]/out_l), out_l)


#%%
inputDim = n_steps_in * len(input_labels) 

model = tf.keras.models.Sequential()
if network == 'mlp':
    for i in range(len(layers_list)):
        if i == 0:     
            model.add(tf.keras.layers.Dense(units = layers_list[i], activation='relu', input_dim=inputDim))
        else:
            model.add(tf.keras.layers.Dense(units = layers_list[i], activation='relu'))
else:
    for i in range(len(layers_list)):
        if len(layers_list) == 1:
            model.add(tf.keras.layers.LSTM(units = layers_list[i], activation='tanh', input_shape=(input_training.shape[1], input_training.shape[2])))
        elif len(layers_list) > 1 and i == 0:     
            model.add(tf.keras.layers.LSTM(units = layers_list[i], activation='tanh', input_shape=(input_training.shape[1], input_training.shape[2]), return_sequences=True))
        elif len(layers_list) > 1 and i == (len(layers_list)-1):
            model.add(tf.keras.layers.LSTM(units = layers_list[i], activation='tanh'))
        else:
            model.add(tf.keras.layers.LSTM(units = layers_list[i], activation='tanh', return_sequences = True))

# Output layer
model.add(tf.keras.layers.Dense(units = n_steps_out, activation = 'linear'))    

# Compilling the network according to the loss_metric
opt = tf.keras.optimizers.Adam(learning_rate=0.001)


model.compile(optimizer = opt, loss = tf.keras.losses.MeanSquaredError(), metrics = [tf.keras.metrics.RootMeanSquaredError()])

es = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 20, verbose = 1)

# Reduce the learnning rate when the metric stop improving.
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 10, min_lr = 1e-7, verbose = 1)

my_dir = os.path.join(".","..","db","models",f"{network}", '_'.join(str(e) for e in layers_list), f"{version}")

mcp =  tf.keras.callbacks.ModelCheckpoint(filepath = my_dir, monitor = 'val_loss', save_best_only= True)

log_dir = os.path.join(".","logs",f"{network}", '_'.join(str(e) for e in layers_list), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
#training and storing the history
history = model.fit(x = input_training,
                        y= output_training,
                        validation_split=0.2, 
                        epochs = 128,
                        batch_size = 512,
                        callbacks = [es,rlr,mcp,tb_callback])

print("training finished!")

# %%
