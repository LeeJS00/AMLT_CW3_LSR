import numpy as np
import csv
import os

def readData(filepath: str = 'data'):
    with open(filepath, 'r', newline='') as f:
        csvReader = csv.reader(f)
        listData = [row for row in csvReader]
    
    # print(listData[0])
    np_listData = np.array(listData, dtype=float)
    return np_listData[:, :-63].T, np_listData[:, -63:]