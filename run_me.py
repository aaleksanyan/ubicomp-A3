from os import listdir
import numpy as np
from importFile import importFile
from featureExtract import createFeatureMatrix

# "first, from EDA data, extract the following features"
# "1. mean, max, minimum EDA level."
# "2. slope of EDA level, using linear regression"
# "3. mean EDA peak height"
# "4. number of EDA peaks"

# We need to create a feature matrix.
# windowSize, windowShift = x, y
# edaFeatureMatrix = np.zeros(0, 7) --- mean, max, min, slop, mean peak height, num peaks, class
# For each thing in EDA_data_csv:
# - if rest, set class = 0. if present, set class = 1.
# - data = importFile(filename)
# - get feature matrix, windowing with windowSize and windowShift
# - add column of "class" to side of matrix
# - concat matrix to edaFeatureMatrix
windowSize = 0
windowShift = 0
edaFeatMatrix = np.zeros((0,7))
for fileName in listdir('EDA_Data_csv'):
    if 'Rest' in fileName:
        label = 0
    else:
        label = 1
    path = 'EDA_Data_csv\\' + fileName
    data = importFile(path)
    featureMatrix = createFeatureMatrix(data, windowSize, windowShift)
    height = featureMatrix.shape[0]
    if(label):
        labelCol = np.ones((height,1))
    else:
        labelCol = np.zeros((height,1))
    featureMatrix = np.concatenate((featureMatrix, labelCol), axis=1)
    edaFeatMatrix = np.concatenate((edaFeatMatrix, featureMatrix), axis=0)


