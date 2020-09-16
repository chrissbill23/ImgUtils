import os
import imageio
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from skimage import transform,io
import multiprocessing as mp

def collect_files(filenames,dirname, gray=False, size=None):
    Xtmp = []
    Ytmp = []
    for filename in filenames:
        Ytmp.append(os.path.basename(dirname))
        img = io.imread(os.path.join(dirname, filename), as_gray=gray)
        if size is not None:
            img = transform.resize(img, size)
        Xtmp.append(img)
    return (Xtmp, Ytmp)
def loadimgdataset(pathmaindir : str, codlabels='ohe', gray=False, size=None, parallel=False):
    encoder = None
    X = []
    Y = []
    tot_works = mp.cpu_count()
    if parallel == False:
        for dirname, _, filenames in tqdm(os.walk(pathmaindir)):
            ris = collect_files(filenames, dirname, gray, size)
            Y += ris[1]
            X += ris[0]
    else:
        with mp.Pool(processes=tot_works) as pool:
            count = 1
            res = []
            for dirname, _, filenames in os.walk(pathmaindir):
                res.append(pool.apply_async(collect_files, args=(filenames, dirname, gray, size)))
                count += 1
            for r in res:
                res2 = r.get(timeout=800)
                X += res2[0]
                Y += res2[1]
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
'''
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
'''
