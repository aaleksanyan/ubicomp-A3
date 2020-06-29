from os import listdir
import numpy as np
from importFile import importFile
from featureExtract import createFeatureMatrix
from sklearn.linear_model import LogisticRegression as logr
from sklearn.metrics import f1_score as f1
import matplotlib.pyplot as plt

# Takes input of a dataType (either 'eda' or 'face')
# Returns feature matrix created with all data files in corresponding folder.
def processData(dataType='eda'):
    if dataType=='eda':
        windowSize, windowShift = 160, 8 # 16Hz
        featureMatrix = np.zeros((0,7))
        # 1 point * 6 feats + 1 class label
        folder = 'EDA_Data_csv'
        skipLines = 8
    elif dataType=="face":
        windowSize, windowShift = 300, 15 # 30Hz
        featureMatrix = np.zeros((0,246)) 
        # 49 points * 5 feats + 1 class label
        folder = "Face_Features"
        skipLines = 1
    else:
        print("Don't know how to deal with this data type: ", dataType)
        return np.zeros((1,1))

    for fileName in listdir(folder):
        # TODO: worth noting - if someone's name had "Rest" or "Present" in it, there could be issues... 
        if 'Rest' in fileName: # If it's a "Rest"
            label = 0
        else: # If it's a "Present"
            label = 1
        path = folder + '\\' + fileName
        data = importFile(path, linesToSkip=skipLines)

        # Create EDA feature matrix
        featureMatrixFile = createFeatureMatrix(data, windowSize, windowShift, dataType=dataType)
        print("Feat matrix file:", featureMatrixFile.shape)

        # Add label column, depending on class label from earlier
        height = featureMatrixFile.shape[0]
        if(label): # 1
            labelCol = np.ones((height,1))
        else: # 0
            labelCol = np.zeros((height,1))
        featureMatrixFile = np.concatenate((featureMatrixFile, labelCol), axis=1)    

        featureMatrix = np.concatenate((featureMatrix, featureMatrixFile), axis=0)

    return featureMatrix

# def getEDA():
#     windowSize, windowShift = 160, 8
#     edaFeatMatrix = np.zeros((0,7)) # mean, max, min, slop, mean peak height, num peaks, class label
#     for fileName in listdir('EDA_Data_csv'):
#         if 'Rest' in fileName:
#             label = 0
#         else: # If it's a "Present"
#             label = 1
#         path = 'EDA_Data_csv\\' + fileName
#         data = importFile(path, linesToSkip=8)

#         # Create EDA feature matrix
#         featureMatrix = createFeatureMatrix(data, windowSize, windowShift)

#         #print("Data matrix shape:", data.shape, label)
#         #print("Resulting feature matrix shape:", featureMatrix.shape)

#         # Add label column, depending on class label from earlier
#         height = featureMatrix.shape[0]
#         if(label): # 1
#             labelCol = np.ones((height,1))
#         else: # 0
#             labelCol = np.zeros((height,1))
#         featureMatrix = np.concatenate((featureMatrix, labelCol), axis=1)

#         # Add to cumulative matrix
#         edaFeatMatrix = np.concatenate((edaFeatMatrix, featureMatrix), axis=0)
        
#     return edaFeatMatrix

# Take in a numpy array and a percentage
# Return a tuple of the array shuffled and split by that percentage
def shuffleAndSplit(arr, percent):
    if percent > 100 or percent < 0:
        return arr
    np.random.shuffle(arr)
    splitPoint = round(len(arr) * (0.01 * percent))  
    arr1 = arr[0:splitPoint, :]
    arr2 = arr[splitPoint:, :]
    return arr1, arr2

#############################
#******functions above******#
#*********`~.***********`~.*#
#****`~. main below `~.*****#
#############################

edaMatrix = processData(dataType='eda')
faceMatrix = processData(dataType='face')

print("Cumulative EDA Feature Matrix:", edaMatrix.shape)
print("Cumulative Face Feature Matrix:", faceMatrix.shape)

# Feature matricies constructed! Next step:
# Rank features individually based on performance metric, such as F score with a binary classifier.
# Use 70% for training and 30% for testing.
# Create bar plot to show F score (y axis) and feature (x axis)
# Use this plot to ascertain the most important feature.

# Task 3 -- Ascertain the best single feature
# For each feature column in eda:
fscores = []
names = []

labelCol = edaMatrix[:, -1].reshape(-1, 1) # Slice returns 1D row vector
for col in range(edaMatrix.shape[1] - 1):
    featCol = edaMatrix[:, 0].reshape(-1, 1) # Slice returns 1D row vector
    oneFeature = np.concatenate((featCol, labelCol), axis=1) # (n,1)*2=(n,2)
    training, test = shuffleAndSplit(oneFeature, 70) 
    model = logr().fit(training[:, :-1].reshape(-1,1), training[:, -1]) # 2D X, 1D y
    y = test[:, -1]
    yhat = model.predict(test[:, :-1].reshape(-1,1)) # Predict wants 2D x. Returns 1D y
    fscore = f1(y, yhat)
    names.append("EDAcol" + str(col))
    fscores.append(fscore)

labelCol = faceMatrix[:, -1].reshape(-1, 1) # Slice returns 1D row vector
for col in range(faceMatrix.shape[1] - 1):
    featCol = faceMatrix[:, 0].reshape(-1, 1) # Slice returns 1D row vector
    oneFeature = np.concatenate((featCol, labelCol), axis=1) # (n,1)*2=(n,2)
    training, test = shuffleAndSplit(oneFeature, 70) 
    model = logr().fit(training[:, :-1].reshape(-1,1), training[:, -1]) # 2D X, 1D y
    y = test[:, -1]
    yhat = model.predict(test[:, :-1].reshape(-1,1)) # Predict wants 2D x. Returns 1D y
    fscore = f1(y, yhat)
    names.append("Facecol" + str(col))
    fscores.append(fscore)    

print(fscores)
ypos = np.arange(len(names))
plt.bar(ypos, fscores, align='center', alpha=0.5)
plt.xticks(ypos, names)
plt.ylabel('fscore')
plt.show()
    