import numpy as np

# Takes in a np array, windowSize, and windowShift to return a feature matrix where every line is a window.
# For EDA, we are just taking the last column and calculating features for that per window. 
# This makes our feature matrix size (n, 6)
# For Face Features, we are taking 5 time domain features for each of the face features.
# There are 32 facial feature columns, so output of matrix size (n, 32*5) 
def createFeatureMatrix(arr, windowSize, windowShift, dataType="eda"):
    # Create cumulative feature matrix
    fm = np.zeros((0,5))
    length = arr.shape[0]
    for index in range(0, length, windowShift):
        end = index + windowSize
        if end > length:
            end = length
        getFeatures(arr[index])

    pass