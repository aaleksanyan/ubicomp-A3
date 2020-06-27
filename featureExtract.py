import numpy as np
from scipy import stats
from scipy.signal import find_peaks


# Takes in a np array, windowSize, windowShift, and dataType to return a feature matrix.
# dataType = 'eda' -- calculating 6 features for the last column. Feature matrix size (n, 6)
# dataType = 'face' -- calculating 5 features for each of 49 columns. Feature matrix size (n, 245)  
def createFeatureMatrix(arr, windowSize, windowShift, dataType="eda"):
    
    # if dataType == 'eda':
    #     # Initialize cumulative feature matrix
    #     fm = np.zeros((0,6))
    #     length = arr.shape[0]
    #     # Start windowing
    #     for index in range(0, length, windowShift):
    #         end = index + windowSize
    #         if end + 1 > length:
    #             end = length - 1
    #         if end-index == 0:
    #             continue
    #         feats = getFeatures(arr[index:end, -1].reshape(end-index,1), {'mean', 'max', 'min', 'slope', 'mph', 'numpeak'})
    #         #print(feats.shape)
    #         fm = np.concatenate((fm, feats.reshape(1,6)), axis=0)
        
    #     return fm
    
    # Initialize cumulative feature matrix. num cols * features
    # Set features of interest
    if dataType == 'face':
        fm = np.zeros((0,245))
        features = {'std', 'slope', 'mean', 'median', 'max'}
    elif dataType == 'eda':
        fm = np.zeros((0,6))
        features = {'mean', 'max', 'min', 'slope', 'mph', 'numpeak'}
    else:
        print('dataType arg was invalid, doofus:', dataType)
        return np.zeros((1,1))

    # Start windowing    
    length = arr.shape[0]
    for index in range(0, length, windowShift):
        # Set endpoint in slice
        end = index + windowSize
        if end + 1 > length:
            end = length - 1 # TODO: Do I need the -1, +1?
        if end-index == 0:
            continue
        
        if dataType == 'face':
            window = arr[index:end, 1:] # Shape is (n, 49)
        if dataType == 'eda':        
            window = arr[index:end, -1].reshape(end-index,1) # Makes it (n, 1) instead of (n, ). A one-col slice just outputs a 1D row vector, but getFeatures wants 2D.
        featureRow = getFeatures(window, features)
        fm = np.concatenate((fm, featureRow), axis=0)

    return fm


# Takes as input 2D np.array matrix and one set of features. Calculates each feature for each column.
# With an input of one column and five features, you would receive a row vector of length five - (1,5)
# With an input of 3 columns and four features, you would receive a row vector of length twelve - (1,12)
# Possible args for features: 'mean', 'median', 'max', 'min', 'slope', 'mph' (mean peak height), 'numpeak' (number of peaks), 'std' (standard deviation)
# The resulting row vector will always be in the above order column by column. [c1-mean, c1-min, c1-mph, c2-mean, c2-min, c2-mph]
def getFeatures(inp, feats):
    featureRow = [] # Initialize where we accumulate features.
    
    # For each column, we need to calculate all the features.
    for col in range(inp.shape[1]):
        data = inp[:,col]
        if len(data) < 2:
            return np.zeros((1,len(feats)))
        # Since sets are O(1) membership checks, they're the optimal DS. But, they may not stay in order.
        # Therefore, this code will always output row features in this order.
        if 'mean' in feats:
            try:
                mean = data.mean()
            except Exception as e:
                mean = 0
                print("Mean calculation error occured.", e)
            featureRow.append(mean)
        if 'median' in feats:
            featureRow.append(np.median(data))
        if 'max' in feats:
            featureRow.append(data.max())
        if 'min' in feats:
            featureRow.append(data.min())
        if 'slope' in feats:
            xvals = np.array(range(len(data))) 
            np.seterr(all="raise")
            try:
                slope, _, _, _, _ = stats.linregress(xvals, data)
            except:
                slope = 0
                print("Error calculating slope on this data:", data)
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
        if 'numpeak' in feats: # Number of peaks
            featureRow.append(len(peaks))
        if 'std' in feats: # Standard deviation
            featureRow.append(data.std())
    
    # This makes featureRow a 2D array instead of a 1D array. (f*c, ) -> (1, f*c)
    return np.array(featureRow).reshape((1, len(featureRow)))