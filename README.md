
# ImgUtils
This module aims to supply functions and classes to upload and preprocess images for machine learning and deep learning tasks.
## Requirements
* Python 3.X
* OpenCV
* Scikit-Learn
* tqm
* Skimage
* Matplotlib
## User API

``` python
loadimgdataset(pathmaindir : str, codlabels :str = 'ohe', gray :bool = False, size :tuple = None, parallel :bool = False):
```
Returns a tuple of size 3 where the first element are the loaded images, the second element are the corresponding encoded labels of the loaded images, and the third element is the label encoding object from scikit-learn.

**pathmaindir :** path to the root directory of the dataset, where the labelled images must be grouped in sub-directories. Each sub-directory must have the name of the label.

**codlabels :**  the encoding of the label. It can be One-Hot-Encoding (value 'ohe'), Binary Encoding (value 'binary') or Integer Encoding (value 'integer').

**gray :**  load the image in gray scale or RGB.

**size :**  new fixed size of all images loaded. If none, each image maintain its original size, otherwise they will be resized according the specified size. 

**parallel :**  whether the task should be parallelized on the available CPUs.

