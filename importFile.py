import numpy as np
# Imports file from CSV to NumPy Array

# Take in path and an optional argument of how many lines to skip in the beginning and parse to return a NumPy array populated with the data.
# Also ensures that data is 'clean' - just regular float64 
def importFile(filePath, linesToSkip=0):
    print("Processing %s..." %(filePath))
    dataCSV = open(filePath)
    if linesToSkip: # If it's >0
        skip = True # Set a flag to say keep reading lines
        lines = 0
    else:
        skip = False 
    data = []
    
    line = dataCSV.readline()

    while line: # While the line is something real
        if skip:
            line = dataCSV.readline()
            lines += 1
            if lines >= linesToSkip: # If we've gotten to/past the number of lines we need to skip
                skip = False
            continue
        
        vals = np.array(line.split(","))
        try: # If can succesfully convert to float, add it to the list
            vals = vals.astype(np.float64)
            data.append(vals)
        except:
            print("Following line did not parse:", vals)
        line = dataCSV.readline()

    dataCSV.close()
    return np.array(data)
