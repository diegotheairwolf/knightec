import numpy as np
import os
import pickle
from PIL import Image

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


def to_images(data):
  """
  Converts time-series data into normalized images
  :param data:
  :return:
  """
  min_val = np.amin(data, axis=1).reshape(-1, 1)
  max_val = np.amax(data, axis=1).reshape(-1, 1)
  P = np.around(((data - min_val) / (max_val - min_val)) * 255)
  # P = data-min_val / (max_val-min_val)
  return P


sample_image = [im.reshape(M, M) for im in to_images(data.X_v2x_train)][0]
new_img = Image.fromarray(sample_image)
new_img.show()

out = new_img.resize((512, 512))
out.show()

print("end")