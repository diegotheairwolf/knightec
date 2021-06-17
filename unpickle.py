import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

# work directory of all data
work_dir = os.getcwd()
rdir = os.path.join(os.path.expanduser(work_dir), 'Datasets')

M = 64
slice_size = M*M
sample_rate = 5000
T = 1.0 / sample_rate
N = (2 - 0) * sample_rate
time = np.linspace(0, 2, N)

# load all data previously pickled (preprocess.py)
filename = 'knightec_100_5000_{}.pickle'.format(M)
with open(os.path.join(rdir, filename), 'rb') as f:
  data = pickle.load(f)


arrays = [data.X_v2x_train, data.X_v2y_train, data.X_v2z_train]
X_train = np.stack(arrays, axis=2)



print("end")