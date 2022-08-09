#%%
from utils import read_dat_file, get_list_of_files, split_sequence
import pandas as pd
import os, joblib, argparse

from sklearn import preprocessing, model_selection
#%%
# Initializing Parser
parser = argparse.ArgumentParser(description ='Softex - PV Power Predection - Dataset Processing Script')
  
# Adding Argument
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

folders = [os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2019"), 
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020"),
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2021")]
    

lst = []
for i in range(len(folders)):
    lst = lst + get_list_of_files(folders[i])

dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)
# %%
#Data preparation
#Removing not a number from the dataset
df_label = df_complete[input_labels]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)

#Normalization
my_dir = os.path.join(".","..","db","saves","norm")
check_folder = os.path.isdir(my_dir)

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(check_folder)

for i in input_labels:
    normalizator = preprocessing.MinMaxScaler(feature_range=(0,1))
    normalizator.fit(df_label[i].values.reshape(-1, 1))
    df_label[i] = normalizator.transform(df_label[i].values.reshape(-1, 1))
    joblib.dump(normalizator, my_dir+'/norm'+str(i)+'.save')


#Splitting the data into training and test data.
trainingData, testData =  model_selection.train_test_split(df_label,test_size = 0.1, shuffle=False)
trainingData = trainingData.reset_index(drop=True)
testData = testData.reset_index(drop=True)


#Splitting the training data into input and output.
inputData, outputData = split_sequence(trainingData, n_steps_in, n_steps_out, input_labels, output_labels)


#MLPs require that the shape of the input portion of each sample is a vector. 
#With a multivariate input, we will have multiple vectors, one for each time step.
if n_steps_in > 1:    
    inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
if n_steps_out > 1:
    outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])

my_dir = os.path.join(".","..","db","data")
check_folder = os.path.isdir(my_dir)

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(check_folder)

pd.DataFrame(inputData).to_csv((my_dir+'/trainingInputData.csv'), index = False)
pd.DataFrame(outputData).to_csv((my_dir+'/trainingOutputData.csv'), index = False)

#Splitting the test data into input and output.
inputData, outputData = split_sequence(testData, n_steps_in, 
                                                n_steps_out, input_labels, output_labels)

#MLPs require that the shape of the input portion of each sample is a vector. 
#With a multivariate input, we will have multiple vectors, one for each time step.
if n_steps_in > 1:    
    inputData = inputData.reshape(inputData.shape[0], inputData.shape[1]*inputData.shape[2])
if n_steps_out > 1:
    outputData = outputData.reshape(outputData.shape[0], outputData.shape[1]*outputData.shape[2])

my_dir = os.path.join(".","..","db","data")
check_folder = os.path.isdir(my_dir)

# If folder doesn't exist, then create it.
if not check_folder:
    os.makedirs(check_folder)

pd.DataFrame(inputData).to_csv((my_dir+'/testInputData.csv'), index = False)
pd.DataFrame(outputData).to_csv((my_dir+'/testOutputData.csv'), index = False)




# %%
