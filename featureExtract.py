import numpy as np
from scipy import stats
from scipy.signal import find_peaks


# Takes in a np array, windowSize, and windowShift to return a feature matrix where every line is a window.
# For EDA, we are just taking the last column and calculating features for that per window. 
# This makes our feature matrix size (n, 6)
# For Face Features, we are taking 5 time domain features for each of the face features.
# There are 32 facial feature columns, so output of matrix size (n, 32*5) 
def createFeatureMatrix(arr, windowSize, windowShift, dataType="eda"):
    
    if dataType == 'eda':
        # Initialize cumulative feature matrix
        fm = np.zeros((0,6))
        length = arr.shape[0]
        # Start windowing
        for index in range(0, length, windowShift):
            end = index + windowSize
            if end + 1 > length:
                end = length - 1
            if end-index == 0:
                continue
            feats = getFeatures(arr[index:end, -1].reshape(end-index,1), {'mean', 'max', 'min', 'slope', 'mph', 'numpeak'})
            #print(feats.shape)
            fm = np.concatenate((fm, feats.reshape(1,6)), axis=0)
        
        return fm

    print('dataType arg was invalid, doofus')

# What would a dream getFeatures() function do, to fulfill our needs in the best way?
# It needs to be able to take inputs of different sizes and return a row of features of predictable length.
# It would also be good to specify exactly what features we want.
# So, it takes in a np.array matrix, and calculates all the specified features for each column.
# With an input of one column and five features, you would receieve a row vector of length five. (1,5) or (5,)
def getFeatures(inp, feats):
    featureRow = [] # Initialize where we accumulate features.
    
    # For each column, we need to calculate all the features.
    for col in range(inp.shape[1]):
        data = inp[:,col]
        if len(data) < 1:
            return np.zeros((1,len(feats)))
        # Mean, max, min, slope, mean peak height, num of peaks
        if 'mean' in feats:
            try:
                mean = data.mean()
            except Exception as e:
                mean = 0
                print("Mean calculation error occured.", e)
            featureRow.append(mean)
        if 'max' in feats:
            featureRow.append(data.max())
        if 'min' in feats:
            featureRow.append(data.min())
        if 'slope' in feats:
            xvals = np.array(range(len(data))) 
            try:
                slope, _, _, _, _ = stats.linregress(xvals, data)
            except:
                slope = 0
                print("Error calculating slope")
            featureRow.append(slope)
        if 'mph' in feats or 'numpeak' in feats:
            # This is here so that we never have to repeat running "find_peaks" if it's needed for 'mph' and 'numpeak'
            peaks, properties = find_peaks(data, height=-np.inf)
        if 'mph' in feats: # Mean peak height
            if properties['peak_heights'].size: # not empty
                mph = properties['peak_heights'].mean()
            else:
                mph = 0
            featureRow.append(mph)
        if 'numpeak' in feats:
            featureRow.append(len(peaks))
        if 'std' in feats:
            featureRow.append(data.std())
    return np.array(featureRow)