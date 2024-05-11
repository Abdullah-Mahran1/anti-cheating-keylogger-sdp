# WARNING: before calling this code, you should have a folder with the passed name,
# the folder should have a file named "Studentkeystrocks.txt" containing the keystrokes
# 
# The code will create the files: data.h5(raw data after being preprocessed) + model.keras(trained ML model)
# ---
# This is The 1D_2xLSTM32 model build and train file. when runing this file you would get at the end the saved traind model file.
# TODO:
# If I want to generate_false_data from users of the system, I should define a way to guarantee that user x isn't trainging on his keystrokes with a "False" label. his keystrokes from the false_data should be execluded 
# DONE - Duration time has -ve values (was error from uController code)
# DONE - Need to deal with empty spaces (  KeyUp 21321) instead of (a KeyUp 21321)
import sklearn.metrics
import sklearn.model_selection
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

train1_data = np.array([])
train1_labels = np.array([])
test1_data = np.array([])
test1_labels = np.array([])
doTrain = True
train_data = np.array([])
train_labels = np.array([])
train0_data = np.array([])
train0_labels = np.array([])
test0_data = np.array([])
test0_labels = np.array([])
input_shape = (30,6)
model = None
batchSize = 32
epochs = 150 # TODO: 150?
callbacks_list = []
fileDir = os.path.dirname(os.path.dirname(__file__))
sample_size = 2000
print(f'--- fileDir: {fileDir} ---')

# from google.colab import drive
# drive.mount('/content/drive')


# # Data loading
# 
# 
# ---
# 
# Description: Load the data from a given directory and aplies Sliding window over it with the given size ws.
# 
# Output: no returns, load in to the global variables the data from the h5 file.
# 
# Algorithm: first we load the rain data from the given h5 file. then we take the first 1000 tupels to be the testing data. for lable 0 and 1 to val0 and val1 respectivly.
# Then we reshape the data to an array of window where every window is of size ws given to the function.
# and we save all the first 1000 in the test variables, bothe data and lables. 
# 

# In order to run this notebook you have to mount your google drive to colab or upload the data sets to colab framework and copy the path of the files.

# In[ ]:
def sliding_window(arr, window_size):
    """
    Applies a sliding window to a numpy array.

    Parameters:
    - arr: Input numpy array of shape (n_samples, n_features)
    - window_size: Size of the sliding window

    Returns:
    - Numpy array with sliding window applied, shape (n_samples, window_size, n_features)
    """
    num_samples,num_features = arr.shape
    print(f'num_samples: {num_samples}')
    # num_features = 6
    result = np.zeros((num_samples, window_size, num_features))

    for i in range(num_samples):
        for j in range(window_size):
            if i - j >= 0:
                result[i, j, :] = arr[i - j, :]

    return result

def generate_false_data(data_len):
    total_lines = 0
    false_data = np.array([])
    print(f'false data collection')
    with h5py.File(os.path.join(os.path.dirname(fileDir),'permenant','training_data.h5'), 'a') as file:
        dataset_names = list(file.keys())
        print(f'dataset_names: {dataset_names}')
        tab_len = int(data_len / (len(dataset_names)-1))
        for dataset_name in dataset_names:
            if dataset_name != fileName:
                dataset = file[dataset_name]
                print("dataset.shape: {dataset.shape}")
                # Get the first 100 data points from the dataset
                false_data = np.append(false_data,dataset[:tab_len])
    false_data = false_data.reshape(-1,input_shape[0],input_shape[1])
    false_labels = np.zeros(len(false_data))
    # false_data = sliding_window(false_data,window_size=input_shape[0])
    return false_data,false_labels



    
    # with h5py.File(os.path.join(os.path.dirname(fileDir),"permenant","false_data.h5"),'r') as h5File:
    #     false_data = h5File.get('train_data')
    #     print(f'max false_data size: {false_data.shape[0]}')
    #     false_data = false_data[:min(int(sample_size/2),false_data.shape[0])]
    #     print(f'but selected false_data size: {false_data.shape[0]}')
    # false_labels = np.zeros(len(false_data))


def preprocess_strokes():
    up = []
    down = []
    inputVector = []
    f = open(os.path.join(fileDir,"Studentkeystrocks.txt"), 'r')
    print(f"FILEDIR: {fileDir}")
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
    loopRange = min(len(rawArr),sample_size)
    print(f'loopRange: {loopRange}, optimal is {sample_size}')
    if loopRange < 100:
        raise ValueError("loopRange must be at least 100")
    for row in range(loopRange):
        for row2 in range(row+1,len(rawArr)):
            try:
                if rawArr[row2][0] == rawArr[row][0] and rawArr[row][1] == 'KeyDown' and rawArr[row2][1] == 'KeyUp':
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
            except Exception as e:
                print(f"skipped char {rawArr[row][0]} \t{e}")
    for myLine in range(len(up)-1):
        k1up = up[myLine]
        k1down = down[myLine]
        k2up = up[myLine+1]
        k2down = down[myLine+1]
        # print(f'#164\tk1up:{k1up}, k1down:{k1down}\n\tk2up:{k2up}, k2down:{k2down}')
        hold1 = k1up[2] - k1down[2] if (k1up[2] - k1down[2]) <3000.0 else 3000.0
        hold2 = k1up[2] - k1down[2] if (k1up[2] - k1down[2]) <3000.0 else 3000.0
        updown = k1up[2] - k1down[2] if (k1up[2] - k1down[2]) <3000.0 else 3000.0
        downdown = k1up[2] - k1down[2] if (k1up[2] - k1down[2]) <3000.0 else 3000.0
        
        #Calculate the times of every feature and save it to the input vactor
        inputVector.append((k1up[0]/255,k2up[0]/255,(hold1)/3000,(hold2)/3000,(downdown)/3000,(updown)/3000)) 
    class_col = np.ones(len(inputVector))
    # if (false_data is not None):
    #     print(f'false_data shape: {len(false_data)} * {len(false_data[0])}, false_data: ')
    #     print(f'false_data[0]: {false_data[0]}, inputVector[0]: {inputVector[0]}')
    #     inputVector.extend(false_data)
    #     print(f'len of class_col before zeros: {len(class_col)}, lst element: {class_col[len(class_col)-1]}')
    #     class_col=np.append(class_col,false_labels)
    #     print(f'len of class_col after zeros: {len(class_col)}, lst element: {class_col[len(class_col)-1]}')
    #     # for i in range(len(inputVector)-1):
    #     # print(f'inputVector shape: {len(inputVector)} * {len(inputVector[i])}')
    inputNP= np.array(inputVector)
    outputNP = sliding_window(inputNP,30) # 30 is the window size
    if doTrain:
        if not os.path.exists(os.path.join(os.path.dirname(fileDir),'permenant','training_data.h5')):
            with h5py.File(os.path.join(os.path.dirname(fileDir),'permenant','training_data.h5'), 'w'):
                print("Created new file.")
        else:
            # with h5py.File(os.path.join(fileDir,".h5"),'a') as h5File:
            with h5py.File(os.path.join(os.path.dirname(fileDir),'permenant','training_data.h5'), 'r+') as h5File:
                # class_col = np.ones(len(inputNP))
                print(f'inputNP len: {len(inputNP)}\n\tclass_col len: {len(class_col)}')
                # print(inputNP)
                print(f'## outputNP len: {len(outputNP)}, class_col len: {len(class_col)} ##')
                print(outputNP)
                # Delete the existing dataset
                print(f'fileName: {fileName}\n\n allFileNames: {h5File}')
                # del h5File['train_data2']
                if fileName in h5File:
                    del h5File[fileName]
                    print(f"Deleted existing dataset '{fileName}'")
                print(outputNP.shape)
                h5File.create_dataset(name=fileName, data=outputNP)
            print(f'returned from 199: {outputNP}')
        return outputNP
                # h5File.create_dataset(name='train_labels', data=class_col,dtype=np.int8)
                # chunks=True,compression='gzip',shuffle=True,
    else:   # not doTrain
        # with h5py.File(os.path.join(fileDir,".h5"),'a') as h5File:
        with h5py.File('testing_data.h5', 'w') as h5File:
            inputNP= np.array(inputVector)
            # class_col = np.ones(len(inputNP))
            print(f'inputNP len: {len(inputNP)}\n\tclass_col len: {len(class_col)}')
            # print(inputNP)
            print(f'## outputNP len: {len(outputNP)}, class_col len: {len(class_col)} ##')
            h5File.create_dataset(name='test_data', data=outputNP,dtype='float64')
            print(f'returned from 211: {outputNP}')
            return outputNP
            # chunks=True,compression='gzip',shuffle=True,

def updateCheatingFile(confidence):
      # Read all lines from the file
  with open("Cheating_Analysis.txt", "r") as file:
      lines = file.readlines()
  # Search for the line with the same fileName
  for i, line in enumerate(lines):
    if fileName in line:
          # Replace the line with the newF data
          if confidence < 0.4:
              most_predicted_class = 0
              percentage_predicted_class = (1 - confidence) * 100
              new_line = f'{fileName},{percentage_predicted_class}% CHEATER\n'
          elif confidence > 0.6:
              most_predicted_class = 1
              percentage_predicted_class = confidence * 100
              new_line = f'{fileName},{percentage_predicted_class}%\n'
          else:
              print('can\'t determine ):')
          lines[i] = new_line

        
          break
  else:
      # If the line is not found, append the new data to the end of the file
      if confidence < 0.4:
          most_predicted_class = 0
          percentage_predicted_class = (1 - confidence) * 100
          new_line = f'{fileName},{percentage_predicted_class}% CHEATER\n'
      elif confidence > 0.6:
          most_predicted_class = 1
          percentage_predicted_class = confidence * 100
          new_line = f'{fileName},{percentage_predicted_class}%\n'
      else:
          print('can\'t determine ):')
      lines.append(new_line)
  
  # Write all lines back to the file
  with open("Cheating_Analysis.txt", "w") as file:
      print("writing to the Cheating_Analysis file")
      file.writelines(lines)
  # Print the result
#   print(f'Accuracy: {accuracy}')
  print("Most of the data points were predicted to be class:", most_predicted_class)
  print("Percentage of data points predicted to be class:", most_predicted_class, ":", percentage_predicted_class, "%")
  print("level of Confidence:", confidence)
  
def update_keystrokes_file():
  lines_to_remove = []
  lines_read = []

    # Read the specified number of lines from the file
  with open(os.path.join(fileDir,"Studentkeystrocks.txt"), 'r') as file:
    for _ in range(sample_size):
        line = file.readline().strip()  # Read line and remove trailing newline characters
        if line:
            lines_read.append(line)
            lines_to_remove.append(line + '\n')  # Add newline character for lines to remove

    # Remove the read lines from the file
  with open(os.path.join(fileDir,"Studentkeystrocks.txt"), 'r+') as file:
    lines = file.readlines()
    file.seek(0)
    for line in lines:
        if line not in lines_to_remove:
            file.write(line)
    file.truncate()

  return lines_read

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

def calculate_metrics(y_true, y_pred,doAll = False):
    # Apply activation to convert logits to probabilities
    y_pred_probabilities = tf.nn.softmax(y_pred)
    # Convert probabilities to class labels
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    # Compute accuracy, precision, recall, and F1 score
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_labels), tf.float32))
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))
    if(doAll):
        precision = sklearn.metrics.precision_score(y_true, y_pred_labels)
        recall = sklearn.metrics.recall_score(y_true, y_pred_labels)
        f1 = sklearn.metrics.f1_score(y_true, y_pred_labels)
        # Compute log loss
        return accuracy, precision, recall, f1, loss
    else:
        return accuracy, loss

def buildModel():
  model = keras.models.Sequential()
  model.add(layers.Conv1D(32,2,activation='relu', input_shape = input_shape))
  model.add(layers.LSTM(32 , return_sequences=True))
  model.add(layers.Dropout(0.5))
#   model.add(layers.LSTM(32 , return_sequences=True))
#   model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  model.add(layers.Dense(2, activation='softmax'))
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optim = keras.optimizers.Adam(learning_rate=0.0001)
  
#   metrics = ['accuracy', 
#           tf.keras.metrics.Precision(), 
#           tf.keras.metrics.Recall(),] 
          # tf.keras.metrics.F1Score(average='weighted')] # Adjust num_classes as per your requirement
  metrics = ['accuracy']
  model.compile(loss=loss, optimizer=optim,metrics=metrics)
  print(model.summary())

  return model

def testModel():
  # global model
  global callbacks_list
  def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
  lrate = LearningRateScheduler(step_decay)
  callbacks_list = [lrate]
  
  
  if(doTrain):
    with tf.device('/device:GPU:0'): #TODO: DO I need to change?
        model.fit(train_data,train_labels,batch_size = batchSize ,validation_split = 0.1, callbacks=callbacks_list,epochs=epochs,shuffle=True)
    #   print(f'Var Lenghts:\n\ttest1_data: {len(test1_data)}\n\ttest0_data: {len(test0_data)}')
    print("[1] Testing for TRUE labels:")
  
    y_pred1 = model.predict(test1_data, batch_size=batchSize)
    accuracy1, loss1 = calculate_metrics(test1_labels, y_pred1)  
    print(f"Performance Metrics for TRUE labels:\n\tLoss1: {loss1}\n\tAccuracy1: {accuracy1}")

    # Determine the most predicted class
    print("[2] Testing for FALSE labels:")
    y_pred0 = model.predict(test0_data, batch_size=batchSize)
    accuracy0, loss0 = calculate_metrics(test0_labels, y_pred0)
    print(f"Performance Metrics for FALSE labels:\n\tLoss0: {loss0}\n\tAccuracy0: {accuracy0}")
    print('[3] Testing for merged labels: ')
    # Merge data and labels
    merged_data = np.concatenate([test0_data, test1_data], axis=0)
    merged_labels = np.concatenate([test0_labels, test1_labels], axis=0)
    # Shuffle merged data
    merged_data, merged_labels = shuffle(merged_data, merged_labels)
    # Calculate metrics for merged predictions
    y_pred_merged = model.predict(merged_data, batch_size=batchSize)
    accuracy_merged, precision_merged, recall_merged, f1_score_merged, loss_merged = calculate_metrics(merged_labels, y_pred_merged,doAll=True)
    # Print performance metrics for merged labels
    print(f"Performance Metrics for Merged labels:\n\tLoss: {loss_merged}\n\tAccuracy: {accuracy_merged}\n\tPrecision: {precision_merged}\n\tRecall: {recall_merged}\n\tF1 Score: {f1_score_merged}")
    # print(f'y_pred1: {y_pred1}, y_pred0: {y_pred0}, y_pred_merged: {y_pred_merged}')
  else: #if not doTrain
    y_pred1 = model.predict(test1_data, batch_size=batchSize)
    count = np.sum(y_pred1[:, 1] > y_pred1[:, 0])
    
    accuracy1 = count / len(y_pred1)
    
    print(f'----Accuracy: {accuracy1}\n\ny_pred1: {y_pred1}')
    updateCheatingFile(accuracy1)
  saveModel()

'''
  model.evaluate(test_data,test_labels,batch_size=batchSize, verbose = 1)
  probability_model = keras.models.Sequential([model,keras.layers.Softmax()])
  predictions = probability_model(val1_data)
  pre0 = predictions[0]
  print(f'pre0 as it is: {pre0}')
  label0 = np.argmax(pre0)
  print(f'label0 as it is: {label0}')
  with open("Cheating_Analysis.txt", "a") as file:
    if (label0 == 1):
        file.write(f'{fileName},{pre0}\n')
    elif (label0 == 0):
        file.write(f'{fileName},{pre0},CHEATER\n')
    else:
        file.write("Unexpected status\n")
'''

def saveModel():
    global model
    try:
        model.save(os.path.join(fileDir,'model.keras'))
        print(f"Model saved successfully as '{fileName}/model.kereas'")
    except Exception as e:
        print(f"Error occurred while saving the model: {e}")

def loadModel(fileName):
    try:
        model = keras.models.load_model(os.path.join(fileDir,'model.keras'))
        print(f"Model loaded successfully from '{fileName}/model.keras'")
        return model
    except FileNotFoundError:
        print(f"File '{fileName}' not found.")
    except Exception as e:
        print(f"Error occurred while loading the model: {e}")
        return None  # Return None if loading fails


# For testing we will take the first 1000 samples and at the end use the to predict the model and test the performance.

# Step [1]: Process parameters <boolean: train/True or test/False> <String: File name>
if __name__ == "__main__":
    print("started here")
    # Parse command line arguments
    try: 
        doTrain_str = str(sys.argv[1])
        doTrain = doTrain_str.lower() == "1"  # Convert the string argument to a boolean
        print(f'doTrain was read to be: {doTrain}')
        fileName = str(sys.argv[2])
        fileDir = os.path.join(str(fileDir),fileName) # = current_directory/fileName
        print(f"FILEDIR: {fileDir}")
    except ValueError:
        print("Error: Invalid arguments. 1st argument must be a boolean, 2nd must be a string")
        sys.exit(1)
# depending on value of doTrain, sample size can be 100 or 2000
sample_size = 1000 if(doTrain) else 100


# [2]: Preprocess the data
# generate_false_data()
data = preprocess_strokes()
labels = np.ones(len(data))
if (doTrain):
    train1_data, test1_data, train1_labels, test1_labels = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=43)
    # print(f'----train1_labels: {min(sample_size,train1_data.shape[0]-1)}\n\ttrain1_labels: {min(sample_size,train1_labels.shape[0]-1)}\n\ttest1_data: {min(sample_size,test1_data.shape[0]-1)}\n\ttest1_labels: {min(sample_size,test1_labels.shape[0]-1)}')
    print(f'train1_data.shape: {train1_data.shape}')
    #get the data of labale 0
    
    train0_data,train0_labels = generate_false_data(len(labels))
    train0_data, test0_data, train0_labels, test0_labels = sklearn.model_selection.train_test_split(train0_data, train0_labels, test_size=0.2, random_state=43)
    print(f'----train0_data: {min(sample_size,len(train0_data)-1)}\n\ttrain0_labels: {min(sample_size,len(train0_labels)-1)}\n\ttest0_data: {min(sample_size,len(test0_data)-1)}\n\ttest0_labels: {min(sample_size,len(test0_labels)-1)}')
    print(f'train0_data.shape: {train0_data.shape}')


    # Combine data and labels
    combined_data = np.concatenate((train1_data, train0_data), axis=0)
    combined_labels = np.concatenate((train1_labels, train0_labels), axis=0)
    train_data,train_labels = shuffle(combined_data,combined_labels)
    # Shuffle combined data and labels in the same order
    # shuffled_indices = np.random.permutation(len(combined_data))
    # train_data = combined_data[shuffled_indices]
    # train_labels = combined_labels[shuffled_indices]
    print("len of Shuffled Data:")
    print(len(train_data))
    print("len of Shuffled Labels:")
    print(len(train_labels))
else:
    test1_data = data[labels == 1.0]
    test1_labels = np.zeros(len(test1_data))
    test0_data = data[labels == 0.0]
    test0_labels = np.zeros(len(test0_data))
    print(f"Test data lengths:\n\ttest1_data: {len(test1_data)}\n\ttest1_labels: {len(test1_labels)}\n\ttest0_data: {len(test0_data)}\n\ttest0_labels: {len(test0_labels)}")

update_keystrokes_file()
print("Done Data preperation..")

# 3: build or load a model
if(doTrain): # for building a model
  model = buildModel()
  saveModel()
else: # for loading a model
  model = loadModel(os.path.join(os.path.dirname(fileDir),fileName)) # TODO: .model or .what file extension??
testModel()