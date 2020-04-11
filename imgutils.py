import os
import imageio
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

def imgdataset(pathmaindir : str, codlabels='ohe', encoder=None):
    X = []
    Y = []
    for dirname, _, filenames in os.walk(pathmaindir):
        for filename in tqdm(filenames):
            Y.append(os.path.basename(dirname))
            X.append(imageio.imread(os.path.join(dirname, filename)))
    X = np.array(X)
    if X.shape[0] > 0:
        if codlabels == 'binary' or codlabels == 'integer':
           encoder = LabelEncoder()
           encoder.fit(Y)
        else:
            if codlabels == 'ohe':
               encoder = LabelBinarizer()
               encoder.fit(Y)
        Y = encoder.transform(Y)
    return X, Y, encoder
    
def imgdatasetreg(pathmaindir : str, substrlabels: list, codlabels='ohe', encoder=None):
    X = []
    Y = []
    for dirname, _, filenames in os.walk(pathmaindir):
        for filename in tqdm(filenames):
            file = os.path.join(dirname, filename)
            s = file.lower()
            for l in substrlabels:
                ls = str(l).lower()
                if ls in s :
                   Y.append(l)
                   X.append(imageio.imread(file))
                   break
    X = np.array(X)
    if X.shape[0] > 0:
        if codlabels == 'binary' or codlabels == 'integer':
           encoder = LabelEncoder()
           encoder.fit(Y)
        else:
            if codlabels == 'ohe':
               encoder = LabelBinarizer()
               encoder.fit(Y)
        Y = encoder.transform(Y)
    return X, Y, encoder
