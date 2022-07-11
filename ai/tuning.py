#%%
import tensorflow as tf
import keras_tuner as kt
import argparse
import pandas as pd
from utils import mae_multi, root_mean_square_error, standard_deviation_error, init_gpus


init_gpus()
#%%
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Hyperparemeter Tunings')

parser.add_argument('--network',
                    type = int,
                    choices={"0", "1"},
                    default="1",
                    help ='Neural network topology which the dataset will be prepared for (0. MLP or 1. LSTM).')
  

parser.add_argument('--input_labels',
                    nargs='+', 
                    default=["Radiacao_Avg", "Temp_Cel_Avg", "Potencia_FV_Avg"],
                    help ='Input features that will be used to make predictions. (TIMESTAMP, Radiacao_Avg,Temp_Cel_Avg, Temp_Amb_Avg,Tensao_S1_Avg,Corrente_S1_Avg, Potencia_S1_Avg, Tensao_S2_Avg, Corrente_S2_Avg, Potencia_S2_Avg, Potencia_FV_Avg, Demanda_Avg,FP_FV_Avg,Tensao_Rede_Avg')

parser.add_argument('--output_labels',
                    type = int,
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

input_labels = args.input_labels
output_labels = args.output_labels
n_steps_in = args.input_steps
n_steps_out = args.output_steps
#%%
#Reading the training and testing data
input_training = pd.read_csv(r'./../db/data/trainingInputData.csv')
input_training = input_training.values
output_training = pd.read_csv(r'./../db/data/trainingOutputData.csv')
output_training = output_training.values
input_test = pd.read_csv(r'./../db/data/testInputData.csv')
input_test = input_test.values
output_test = pd.read_csv(r'./../db/data/testOutputData.csv')
output_test = output_test.values

if network == 'lstm':
    in_l = len(input_labels)
    out_l = len(output_labels)
    input_training = input_training.reshape(input_training.shape[0], int(input_training.shape[1]/in_l), in_l)
    output_training = output_training.reshape(output_training.shape[0], int(output_training.shape[1]/out_l), out_l)

#%%
def build_mlp_model(hp):
    model = tf.keras.Sequential()
    
    for i in range(hp.Int('num_layers', 1, 8)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(tf.keras.layers.Dense(n_steps_out, activation='linear'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    
    return model

def build_lstm_model(hp):
    model = tf.keras.Sequential()
    
    
    model.add(tf.keras.layers.LSTM(hp.Int('input_unit',min_value=32,max_value=256,step=32), activation = 'tanh', return_sequences=True, input_shape=(input_training.shape[1], input_training.shape[2])))
    
    for i in range(hp.Int('n_layers', 1, 2)):
        model.add(tf.keras.layers.LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=256,step=32), activation = 'tanh', return_sequences=True))
    
    model.add(tf.keras.layers.LSTM(hp.Int('last_layer_units',min_value=32,max_value=256,step=32), activation = 'tanh'))
    
    model.add(tf.keras.layers.Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    
    model.add(tf.keras.layers.Dense(n_steps_out, activation='linear'))
    
    model.compile(loss='mean_absolute_error', metrics=['mean_absolute_error'], optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-3, 1e-4])))

    return model

if network == 'mlp':
    tuner = kt.BayesianOptimization(build_mlp_model, objective='val_mean_absolute_error', max_trials=20, executions_per_trial=1,
                     directory='./logs/tuning/',project_name='mlp_param')
else:
    tuner = kt.BayesianOptimization(build_lstm_model, objective='val_mean_absolute_error', max_trials=20, executions_per_trial=1,
                     directory='./logs/tuning/',project_name='lstm_param')

es = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', min_delta = 1e-9, patience = 10, verbose = 1)


#%%
tuner.search_space_summary()

#%%
tuner.search(input_training, output_training,epochs=128, batch_size = 512,
     validation_split=0.2, callbacks=[es], verbose=1)

#%%
tuner.results_summary()

#%%
bayes_opt_model_best_model = tuner.get_best_models(num_models=3)
model = bayes_opt_model_best_model[0]

