#%% Import packages
import os
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
def init_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            #Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            #Memory growth must be set before GPUs have been initialized
            print(e)
#%%
def read_dat_file(filename):
    """
    Read .dat file, discards some row headers and returns appropriate values.

    Parameters
    ----------
    filename : string with path and filename do .dat file

    Returns
    -------
    df : pandas.DataFrame
        A pandas dataframe contatining the data.
    """
    df = pd.read_csv(filename, skiprows=3)
    df_aux = pd.read_csv(filename, header=1)
    df.columns = df_aux.columns

    cols_to_drop = ['RECORD', 'Excedente_Avg', 'Compra_Avg']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop([col], axis=1)

    for column in df.columns:
        if column != "TIMESTAMP":
            df[column] = df[column].astype('float')
    # Drop column 'RECORD' (if present) because from june 2019 is is no longer used
    return df

def get_list_of_files(folder):
    """
    Return a list of *.dat files inside the subfolders of folder 'folder'.

    Parameters
    ----------
    folder : string with path to root folder

    Returns
    -------
    lst : list
        A list containing all *.dat file strings
    """
    lst = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            complete_filename = os.path.join(root, name)
            # print(complete_filename)
            lst.append(complete_filename)
        for name in dirs:
            complete_filename = os.path.join(root, name)
            # print(complete_filename)
            lst.append(complete_filename)

    lst.sort()
    return [x for x in lst if '.dat' in x]

#%%
#Preprocessing Data

def split_sequence(sequence, n_steps_in, n_steps_out,input_labels,output_labels):
    """
    Splits a univariate sequence into samples
    Parameters
    ----------
    data : Series
        Series containing the data sequences you want to divide into samples of inputs and outputs for the model prediction.
    n_steps_in : Integer
        Number of past data which will be used in a single input samples.
        How many minutes back we use to predict the next n_steps_out minutes.
    n_steps_out : Integer
        Number of future data which will represent a single output samples.
        How many minutes we will want to predict.
    input_labels : List
        List of String containnning the labels of the input of the model 
    output_labels : List
        List of String containnning the labels of the output of the model 
    Returns
    -------
    X,y : Numpy Arrays
        X represents the inputs samples and y the outputs samples.

    """
    X, y = list(), list()
    input_sequence = sequence[input_labels]
    output_sequence = sequence[output_labels]
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = input_sequence[i:end_ix], output_sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

#%%
#Evaluation functions
def mae_multi(y_true, y_pred):
    """
    Mean Absolute Square (RMS) error between real Avarage PV Power and the predictions    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    mae : EagerTensor
    """
    #axis=0 to calculate for eache sample.
    return tf.keras.backend.mean(tf.keras.backend.abs(tf.math.subtract(y_true, y_pred)), axis =0)

def root_mean_square_error(y_true, y_pred):
    """
    Root Mean Square (RMS) error between real Avarage PV Power and the predictions    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    rms : EagerTensor
    """
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(tf.math.subtract(y_true, y_pred)), axis =0))

def standard_deviation_error(y_true, y_pred):
    """
    Standart Deviation of the error between real Avarage PV Power and the prediction    
    ----------
    y_true : List
        List containing the real outputs.
    y_pred : List
        List containing the predictions outputs of the network.
    
    Returns
    -------
    rms : EagerTensor
    """
    return tf.keras.backend.std(tf.math.subtract(y_true, y_pred), axis =0)

