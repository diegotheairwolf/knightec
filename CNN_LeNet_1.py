import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# work directory of all data
work_dir = os.getcwd()
rdir = os.path.join(os.path.expanduser(work_dir), 'Datasets')

M = 16
slice_size = 2400
sample_rate = 12000
T = 1.0 / sample_rate
N = (2 - 0) * sample_rate
time = np.linspace(0, 2, N)

# load all data previously pickled (preprocess.py)
filename = 'CWRU.pickle'
with open(os.path.join(rdir, filename), 'rb') as f:
  data = pickle.load(f)

# # load 48k 1730 data previously pickled (preprocess.py)
# filename = 'cwru_48k_1730.pickle'
# with open(os.path.join(rdir, filename), 'rb') as f:
#   data = pickle.load(f)

# get 12kHz data
data_12kHz = [data[i] for i in range(0, 4)]
# get 48kHz data
data_48kHz = [data[i] for i in range(8, 12)]

print(data_12kHz[0].X[15][0])

# get normal data per frequency
data_12normal = [data_12kHz[i].X[15] for i in range(0, 4)]
data_48normal = [data_48kHz[i].X[13] for i in range(0, 4)]

# hp indexes
rpm1797 = 0
rpm1772 = 1
rpm1750 = 2
rpm1730 = 3
# label indexes
ball7 = 0
ir7 = 1

# split by load
data_12normal_1797 = data_12normal[rpm1797]
data_12normal_1772 = data_12normal[rpm1772]
data_12normal_1750 = data_12normal[rpm1750]
data_12normal_1730 = data_12normal[rpm1730]
print(data_12normal_1797.shape)
data_48normal_1797 = data_48normal[rpm1797]
data_48normal_1772 = data_48normal[rpm1772]
data_48normal_1750 = data_48normal[rpm1750]
data_48normal_1730 = data_48normal[rpm1730]
print(data_48normal_1797.shape)

# get faulty data 12k 1797 load and 0,07" ball fault
data_12ball_007 = data_12kHz[rpm1797].X[ball7]
print(data_12ball_007.shape)

# get first 10 seconds = 120000 samples and split into sub-signals of 0.2s = 2400 points
data_12normal_1797_samples = data_12normal_1797[:120000]
print(data_12normal_1797_samples.shape)
data_12ball_007_samples = data_12ball_007[:120000]
print(data_12ball_007_samples.shape)

fft = tf.signal.rfft(data_12normal_1797_samples)

f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(data_12normal_1797_samples)
print(np.abs(fft).shape)
print(f_per_dataset.shape)
print(n_samples_h)

fft_yf = np.abs(fft[0:n_samples_h//2])
fft_abs = np.abs(fft)

plt.step(f_per_dataset, np.abs(fft))
plt.xscale('log')
plt.title("All Normal Data - 12kHz")
_ = plt.xlabel('Frequency (log scale)')

fft = tf.signal.rfft(data_12ball_007_samples)
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(data_12ball_007_samples)
print(np.abs(fft).shape)
print(f_per_dataset.shape)
print(n_samples_h)
plt.step(f_per_dataset, np.abs(fft))
plt.xscale('log')
plt.title("All 0,07 Ball Data - 12kHz")
_ = plt.xlabel('Frequency (log scale)')


# 3. create clips of normal/faulty data and label respectively

# Split data for training and validation
normal_data = np.zeros((0, slice_size))
anomalous_data = np.zeros((0, slice_size))

idx_last = -(data_12normal_1797_samples.shape[0] % slice_size)
normal_data = data_12normal_1797_samples.reshape(-1, slice_size)
print(normal_data.shape)

idx_last = -(data_12ball_007_samples.shape[0] % slice_size)
anomalous_data = data_12ball_007_samples.reshape(-1, slice_size)
print(anomalous_data.shape)

normal_labels = np.ones(normal_data.shape[0])
anomalous_labels = np.zeros(anomalous_data.shape[0])
print(normal_labels.shape)
print(anomalous_labels.shape)

print(normal_data[0].shape)


fft = tf.signal.rfft(normal_data[0])
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(normal_data[0])
print(np.abs(fft).shape)
print(f_per_dataset.shape)
print(n_samples_h)
plt.step(f_per_dataset, np.abs(fft))
plt.xscale('log')
plt.title("Normal Data Sample # {}- 12kHz".format(0))
_ = plt.xlabel('Frequency (log scale)')


ffts_normal = [tf.signal.rfft(normal_data[i]) for i in range(len(normal_data))]
f_per_dataset = np.arange(0, len(ffts_normal[0]))
n_samples_h = len(ffts_normal)

print(len(ffts_normal[0]))
print(np.abs(ffts_normal[0]).shape)
print(f_per_dataset.shape)
print(n_samples_h)


ffts_anomalous = [tf.signal.rfft(anomalous_data[i]) for i in range(len(anomalous_data))]
f_per_dataset = np.arange(0, len(ffts_anomalous[0]))
n_samples_h = len(ffts_anomalous)

print(len(ffts_anomalous[0]))
print(np.abs(ffts_anomalous[0]).shape)
print(f_per_dataset.shape)
print(n_samples_h)

# Combine normal and anomalous data and shuffle.

all_data = np.concatenate((ffts_normal, ffts_anomalous))
print(all_data.shape)
all_labels = np.concatenate((normal_labels, anomalous_labels))
print(all_labels.shape)
print(all_labels[:5])

# https://stackoverflow.com/questions/43229034/randomly-shuffle-data-and-labels-from-different-files-in-the-same-order
idx = np.random.permutation(len(all_data))
all_data, all_labels = all_data[idx], all_labels[idx]
print(all_data.shape)
print(all_labels.shape)
print(all_labels[:5])


# Split into train/test.

train_data, test_data, train_labels, test_labels = train_test_split(
    all_data, all_labels, test_size=0.2, random_state=21)

print(train_data.shape)
print(test_data.shape)


# Normalize the data to [0, 1] to improve training accuracy

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)







# Split data for training and validation
normal_data = np.zeros((1, slice_size))
anomalous_data = np.zeros((1, slice_size))

idx_last = -(data_12normal_1797_samples.shape[0] % slice_size)
normal_data = data_12normal_1797_samples[:idx_last].reshape(-1, slice_size)
print(normal_data.shape)

idx_last = -(data_12ball_007_samples.shape[0] % slice_size)
anomalous_data = data_12ball_007_samples[:idx_last].reshape(-1, slice_size)
print(anomalous_data.shape)

normal_labels = np.ones(normal_data.shape[0])
anomalous_labels = np.zeros(anomalous_data.shape[0])
print(normal_labels.shape)
print(anomalous_labels.shape)









# get only normal data from all loads
data_normal = [data[i].X[15] for i in range(0, 4)]
# split by load
data_normal_1797 = data_normal[0]
data_normal_1772 = data_normal[1]
data_normal_1750 = data_normal[2]
data_normal_1730 = data_normal[3]
# X_train = [data[i].X_train for i in range(0, 4)]
# X_test = [data[i].X_test for i in range(0, 4)]
# y_train = [data[i].y_train for i in range(0, 4)]
# y_test = [data[i].y_test for i in range(0, 4)]











fft = tf.signal.rfft(data_normal_1797)
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(data_normal_1797)
print(np.abs(fft).shape)
print(f_per_dataset.shape)
print(n_samples_h)

plt.step(f_per_dataset, np.abs(fft))
plt.xscale('log')
_ = plt.xlabel('Frequency (log scale)')
plt.show()

# Split data for training and validation
train_data = np.zeros((0, slice_size))
test_data = np.zeros((0, slice_size))

idx_last = -(data_normal_1797.shape[0] % slice_size)
clips = data_normal_1797[:idx_last].reshape(-1, slice_size)
n = clips.shape[0]
n_split = int(3 * n / 4)
train_data = np.vstack((train_data, clips[:n_split]))
test_data = np.vstack((test_data, clips[n_split:]))

print(train_data.shape)
print(test_data.shape)



# Normalize the data to [0,1] to improve training accuracy.
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


# Plot a one sample
plt.grid()
plt.plot(np.arange(train_data[0].shape[0]), train_data[0])
plt.title("Normal Vibration Data 1797RPM Sample")
plt.show()


# Build the model
#
# After training and evaluating the example model, try modifying the size and number of layers to build an understanding for autoencoder architectures.
#
# Note: Changing the size of the embedding (the smallest layer) can produce interesting results. Feel free to play around with that layer size.


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])  # Smallest Layer Defined Here

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(256, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

# Train the model
#
# Notice that the autoencoder is trained using only the normal ECGs, but is evaluated using the full test set.
history = autoencoder.fit(train_data, train_data,
          epochs=20,
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)



# # print("Plot first slice")
# x = np.arange(0, data.X_train.shape[1])
# plt.title("First time slice")
# plt.xlabel("x axis - time")
# plt.ylabel("y axis - vibration")
# plt.plot(x, data.X_train[0])
# plt.show()

# # print("Plot everything")
# vib_data = np.asarray(data.X[0])
# x = np.arange(0, vib_data.size)
# print(vib_data)
# plt.title("All time slices")
# plt.xlabel("x axis - time")
# plt.ylabel("y axis - vibration")
# plt.plot(x, vib_data)
# plt.show()

# # Compute and plot the spectogram of the first time slice
# freqs, times, spectrogram = signal.spectrogram(data.X_train[0])
# plt.figure(figsize=(10, 4))
# plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
# plt.title('Spectrogram')
# plt.ylabel('Frequency band')
# plt.xlabel('Time window')
# plt.tight_layout()
# plt.show()


# # Plot only Record 125DE (48k, 0.007 in. drive end ball fault, 1730 rpm).
# N = len(data.X[0])
# time_axis = len(data.X[0])/48000
# x = np.arange(0,len(data.X[0]))
# plt.title("Record 125DE (48k, 0.007 in. drive end ball fault, 1730 rpm).")
# plt.xlabel("x axis - time")
# # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
# plt.ylabel("y axis - vibration")
# plt.plot(x[:4096], data.X[0][:4096])
# plt.show()


# Compute and plot the FF
# yf = fft(data.X[0])
# xf = fftfreq(N, T)[:N//2]
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()


# ----------------------------------------------

# Use DataFrame for better manipulation
df = pd.DataFrame(data.X[0], columns=[data.labels[0]])
df.index.name = "Cycles"
# print(df)




# ------------------------------------------------
# A. Signal to Image Conversion


# used for just 1 image for testing purposes
def to_image(df, M):
    """
    single image conversion from dataframe[0:M*M] to M*M ndarray
    :param df:
    :param M:
    :return:
    """
    P = df.values.reshape(M, M)
    P = np.round((P-np.min(df.values)/(np.max(df.values)-np.min(df.values)))*255)
    return P

# Convert dataframe element to image for testing purposes
data_to_convert = df[0:(M*M)]
data_image = to_image(data_to_convert, M)
plt.imshow(data_image, cmap='gray')
plt.imshow(data_image)
plt.show()


def to_images(data):
    """
    Converts time-series data into normalized images
    :param data:
    :return:
    """
    min_val = np.amin(data, axis=1).reshape(-1, 1)
    max_val = np.amax(data, axis=1).reshape(-1, 1)
    P = np.around(((data-min_val) / (max_val-min_val)) * 255)
    return P


def show_images(images, labels):
    """
    Creates a collage all class images
    :param images: all images data. ndarray of 64x64 elements
    :param labels: labels of images. 0-13
    :return: none
    """

    image_index, unique_img = [], []
    for i in range(0, 14):
        image_index.append(labels.index(i))
        unique_img.append(images[labels.index(i)].squeeze(axis=2))

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 7),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i, (ax, im) in enumerate(zip(grid, unique_img)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(data.labels[i])

    plt.show()


# convert all time-series slices to images
train_X = np.array([im.reshape(64, 64) for im in to_images(data.X_train)])
test_X = np.array([im.reshape(64, 64) for im in to_images(data.X_test)])

# add channel for grayscale
train_X = train_X[..., np.newaxis]
test_X = test_X[..., np.newaxis]

#display some of the images
show_images(train_X, data.y_train)


def OneHot(y):
    """
        input:  y (n) = labels
        output: onehot (Kxn) = one-hot representation of labels
    """
    onehot = np.zeros((14, len(y)))
    for idx, val in enumerate(y):
        onehot[val][idx] = 1
    # print("oneshot {}".format(onehot))
    return onehot

# convert labels to one-hot encoding
train_y = OneHot(data.y_train).swapaxes(1, 0)
test_y = OneHot(data.y_test).swapaxes(1, 0)


# ------------------------------------------------
# C. Zero-Padding Calculation

def cal_zero_padding(M, N, F, S):
    """
    The number of padding on left PL and on right PR can be calculated by the following equations.
    :param M: input size
    :param N: output size
    :param F: filter width
    :param S: stride
    :return: PL - left padding
            PR - right padding

    """
    N = np.ceil(M/S)
    PT = (N-1) * S + F - M
    PL = np.floor(PT/2)
    PR = PT - PL

    return PT, PL, PR



# ------------------------------------------------
# B. CNN Model

def get_compiled_model():
    """
    model based on LeNet-5
    activation function is tanh
    :return:
    """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=5, input_shape=(M, M, 1), activation="tanh", padding="same", name="L1"))
    model.add(layers.MaxPool2D(pool_size=2, name="L2"))
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation="tanh", padding="same", name="L3"))
    model.add(layers.MaxPool2D(pool_size=2, name="L4"))
    model.add(layers.Conv2D(filters=128, kernel_size=3, activation="tanh", padding="same", name="L5"))
    model.add(layers.MaxPool2D(pool_size=2, name="L6"))
    model.add(layers.Conv2D(filters=256, kernel_size=3, activation="tanh", padding="same", name="L7"))
    model.add(layers.MaxPool2D(pool_size=2, name="L8"))
    model.add(layers.Flatten())
    model.add(layers.Dense(2560, activation='tanh'))
    model.add(layers.Dense(256, activation='tanh'))
    model.add(layers.Dense(14, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='SGD',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mean_squared_error'])

    return model

model = get_compiled_model()

# create checkpoints to save model after each iteration to be used later
# https://www.tensorflow.org/tutorials/keras/save_and_load

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 200

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=5*batch_size)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

model.fit(train_X, train_y, epochs=5, callbacks=[cp_callback], batch_size=batch_size)
# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
# current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)
model_name = timestamp
model.save('{}.h5'.format(timestamp))


# The evaluate() method - gets the loss statistics
model.evaluate(test_X, test_y, batch_size=batch_size)
# returns: loss: 0.0022612824104726315

target = model.predict(test_X)


print("debug")

