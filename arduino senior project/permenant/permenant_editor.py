from sklearn.model_selection import validation_curve
from tensorflow.keras.losses import binary_crossentropy
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
import os
import math
from keras.callbacks import LearningRateScheduler
import ast
import h5py
import sys
import joblib
import KeyMap

train_data = np.array([])
train_labels = np.array([])
val0_data = np.array([])
val0_labels = np.array([])
val1_data = np.array([])
val1_labels = np.array([])
input_shape = (30,6)
model = None
batchSize = 32
epochs = 150
callbacks_list = []
fileDir = os.path.dirname(__file__)
print(f'--- fileDir: {fileDir} ---')

def sliding_window(arr, window_size):
    """
    Applies a sliding window to a numpy array.

    Parameters:
    - arr: Input numpy array of shape (n_samples, n_features)
    - window_size: Size of the sliding window

    Returns:
    - Numpy array with sliding window applied, shape (n_samples, window_size, n_features)
    """
    num_samples, num_features = arr.shape
    result = np.zeros((num_samples, window_size, num_features))

    for i in range(num_samples):
        for j in range(window_size):
            if i - j >= 0:
                result[i, j, :] = arr[i - j, :]

    return result



def preprocess_strokes():
    up = []
    down = []
    inputVector = []
    f = open(os.path.join(fileDir,"Studentkeystrocks.txt"), 'r')
    rawFile = f.read()
    rawLines = rawFile.split("\n") # split the data into a list that every cell in the list is a row in the dataset
    print(f'Studentkeystrocks file in {fileDir}, has {len(rawLines)} lines')
    rawArr = []
    for myLine in rawLines:
        rawArr.append(myLine.split(" "))
    # x.remove(x[len(x)-1])
    print(f'x.shape: {len(rawArr)} * {len(rawArr[0])} or * {len(rawArr[1])}\n x[-1] = {rawArr[len(rawArr)-1]}')
    # print(x)
    # print('\n\n')
    loopRange = min(len(rawArr),500)
    print(f'loopRange: {loopRange}, optimal is 500')
    for row in range(loopRange):
        # print(f'row: {rawArr[row]}')
        for row2 in range(row+1,len(rawArr)):
            # print(f'row2: {rawArr[row2]}')
            if rawArr[row2][0] == rawArr[row][0] and rawArr[row][1] == 'KeyUp' and rawArr[row2][1] == 'KeyDown':
                # these 4 lines can be deleted (mahran)
                if str(rawArr[row][0]).find(',') != -1:
                    rawArr[row][0] = str(rawArr[row][0]).replace(',', '')
                if str(rawArr[row2][0]).find(',') != -1:
                    rawArr[row2][0] = str(rawArr[row2][0]).replace(',', '')
                downcon = [KeyMap.virtualKeyMap[str(rawArr[row][0]).upper()],rawArr[row][1],int(rawArr[row][2])] #check the ID of the key in the keyMap file
                down.append(downcon)
                # print(f'--downCon:{downcon}')
                upcon = [KeyMap.virtualKeyMap[str(rawArr[row2][0]).upper()],rawArr[row2][1],int(rawArr[row2][2])] #check the ID of the key in the keyMap file
                up.append(upcon)
                break
    for myLine in range(len(up)-2):
        k1up = up[myLine]
        k1down = down[myLine]
        k2up = up[myLine+1]
        k2down = down[myLine+1]

        #Calculate the times of every feature and save it to the input vactor
        inputVector.append((k1up[0]/255,k2up[0]/255,(k1up[2] - k1down[2])/1000,(k2up[2] - k2down[2])/1000,(k2down[2] - k1down[2])/1000,(k2down[2] - k1up[2])/1000)) 
        class_col = np.ones(len(inputVector)-1)
    with open(os.path.join(fileDir,"data.txt"), "w") as txtFile:
        txtFile.write(str(inputVector))
    with h5py.File(os.path.join(fileDir,"data.h5"),'w') as h5File:
        inputNP= np.array(inputVector)
        print(class_col)
        print(inputNP)
        outputNP = sliding_window(inputNP,30) # 30 is the window size
        print(f'## outputNP len: {len(outputNP)}, class_col len: {len(class_col)} ##')
        h5File.create_dataset(name='train_data', data=outputNP,dtype=np.float64)
        h5File.create_dataset(name='train_labels', data=class_col,dtype=np.int8)
        # chunks=True,compression='gzip',shuffle=True,

preprocess_strokes()