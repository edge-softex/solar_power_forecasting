# %%
import os
import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from utils import *
#%%
def baseline_mean(df_label, time_shift, amount, label):
    y = []
    y_hat = []

    for i in range(time_shift*amount, np.size(df_label)-1):
        y.append(df_label[label][i])
        aux = []
        for day in range(1, amount+1):
            aux.append(df_label[label][i-(time_shift*day)])
        y_hat.append(np.array(aux).mean())
    
    return np.array(y), np.array(y_hat)

def baseline_one_before(input_test, output_test, normalizator):

    x = normalizator.inverse_transform(input_test)
    y = normalizator.inverse_transform(output_test)

    y_hat = []
    for i in range(len(y)):
        y_hat.append(np.repeat(x[i][-1],5))
    
    return np.array(y), np.array(y_hat)

def linear_regression(training_input, training_output, test_input, test_output):
    model = LinearRegression()
    model.fit(training_input, training_output)
    y_hat = model.predict(test_input)

    return np.array(test_output), np.array(y_hat)
#%%

folders = [os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2019"), 
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2020"),
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2021"),
            os.path.join(".","..","db","Dados Sistema 5 kW","Ano 2022")]
    
input_labels = ["Radiacao_Avg", "Temp_Cel_Avg", "Potencia_FV_Avg"]
output_labels = ["Potencia_FV_Avg"]

lst = []
for i in range(len(folders)):
    lst = lst + get_list_of_files(folders[i])

#%%
dfs = (read_dat_file(f) for f in lst)
df_complete = pd.concat(dfs, ignore_index=True)


#%%
# MEAN BASELINE 
df_label = df_complete[output_labels]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)

trainingData, testData =  model_selection.train_test_split(df_label,test_size = 0.2, shuffle=False)
trainingData = trainingData.reset_index(drop=True)
testData = testData.reset_index(drop=True)

time_shift = 1440
mean_days = 5

y, y_hat = baseline_mean(testData, time_shift, mean_days,output_labels[0])
 
#%%
# ONE MINUTE B4 BASELINE 
input_test = pd.read_csv(r'./../db/data/testInputData.csv')
input_test = input_test.values

output_test = pd.read_csv(r'./../db/data/testOutputData.csv')
output_test = output_test.values

normalizator = joblib.load(r'./../db/norm/normPotencia_FV_Avg.save')

y, y_hat = baseline_one_before(input_test, output_test, normalizator)

#%%
#LINEAR REGRESSION
df_label = df_complete[input_labels]
df_label = df_label.dropna()
df_label = df_label.reset_index(drop=True)

n_steps_in = 120
n_steps_out = 5

trainingData, testData =  model_selection.train_test_split(df_label,test_size = 0.2, shuffle=False)
trainingData = trainingData.reset_index(drop=True)
testData = testData.reset_index(drop=True)

training_input, training_output =  split_sequence(trainingData, n_steps_in, n_steps_out, input_labels, output_labels)
test_input, test_output = split_sequence(testData, n_steps_in, n_steps_out, input_labels, output_labels)

training_input = training_input.reshape(training_input.shape[0], training_input.shape[1]*training_input.shape[2])
training_output = training_output.reshape(training_output.shape[0], training_output.shape[1]*training_output.shape[2])

test_input = test_input.reshape(test_input.shape[0], test_input.shape[1]*test_input.shape[2])
test_output = test_output.reshape(test_output.shape[0], test_output.shape[1]*test_output.shape[2])

y, y_hat = linear_regression(training_input, training_output, test_input, test_output)


#%%
baseline_mae = np.array(mae_multi(y, y_hat))
baseline_rmse = np.array(root_mean_square_error(y, y_hat))
baseline_std = np.array(standard_deviation_error(y, y_hat))

#%%
print("Tests evaluation:")
print("Mean Absolute Error (MAE): {:.3f}".format(baseline_mae.mean()))
print("Root Mean Square Error (RMS): {:.3f}".format(baseline_rmse.mean()))
print("Standart Deviation: {:.3f}".format(baseline_std.mean()))
# %%
