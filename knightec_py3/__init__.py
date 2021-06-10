import os
import glob
import errno
import random
import urllib.request as urllib
import numpy as np
from scipy.io import loadmat

import re
import sys
from shutil import copyfile
import csv
import pandas as pd
from tqdm import tqdm


class KNIGHTEC:

    def __init__(self, exp, rpm, sr, length):

        self.sensors = ["v1", "v2x", "v2y", "v2z", "t1", "t2", "c1", "c2", "c3"]
        self.labels = ["Bearing1", "Healthy", "HighSpeed", "Misalignment", "Bearing1+Shaft"]
        self.nclasses = len(self.labels)  # number of classes

        if exp not in self.labels:
            print("wrong experiment name: {}".format(exp))
            exit(1)
        if rpm not in ("100", "75", "50"):
            print("wrong rpm value: {}".format(rpm))
            exit(1)
        # work directory of all data
        work_dir = os.getcwd()
        rdir = os.path.join(os.path.expanduser(work_dir), 'Datasets/Knightec')

        fmeta = os.path.join(os.path.dirname(__file__), 'metadata.txt')

        walk_dir = os.path.join(os.getcwd(), 'data')
        filepaths = self._create_metadata(fmeta, walk_dir)

        # create list of files that will be used in this experiment
        all_lines = open(fmeta).readlines()
        lines = []
        fpaths = []
        for idx, line in enumerate(all_lines):
            filename = os.path.split(line)[1]
            l = re.split('%_|__|_|\.|\n', filename)
            if l[0] == rpm and l[2] == sr:
                lines.append(l)
                fpaths.append(filepaths[idx])

        # adding exp and rpm data for keeping track
        self.exp = exp
        self.rpm = rpm
        self.sr = sr
        self.length = length  # sequence length
        self._load_and_slice_data(rdir, lines, fpaths)
        # shuffle training and test arrays
        self._shuffle()


    def _mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _create_metadata(self, fmeta, walk_dir):
        filenames = []
        open(fmeta, 'w').close()
        # https://stackoverflow.com/questions/2212643/python-recursive-folder-read
        for root, subdirs, files in os.walk(walk_dir):
            # skip root folder where only readme.txt file exists
            if root == walk_dir:
                continue
            # print('metadata_file_path = ' + fmeta)
            with open(fmeta, 'ab') as list_file:
                # for subdir in subdirs:
                #     print('\t- subdirectory ' + subdir)
                for filename in files:
                    if filename == ".DS_Store":
                        continue
                    file_path = os.path.join(root, filename)
                    filenames.append(file_path)
                    # print('\t- file %s (full path: %s)' % (filename, file_path))
                    list_file.write(file_path.encode('utf-8'))
                    list_file.write(b'\n')
        return filenames

    def _download(self, fpath, link):
        print("Downloading from '{}' to '{}'".format(link,fpath))
        urllib.URLopener().retrieve(link, fpath)

    def _load_and_slice_data(self, rdir, infos, filepaths):
        self.y_train = []
        self.y_test = []

        self.files = []

        self.X_v1 = []
        self.X_v2x = []
        self.X_v2y = []
        self.X_v2z = []
        self.X_c1 = []
        self.X_c2 = []
        self.X_c3 = []
        self.X_t1 = []
        self.X_t2 = []

        self.X_v1_train = np.zeros((0, self.length))
        self.X_v1_test = np.zeros((0, self.length))
        self.X_v2x_train = np.zeros((0, self.length))
        self.X_v2x_test = np.zeros((0, self.length))
        self.X_v2y_train = np.zeros((0, self.length))
        self.X_v2y_test = np.zeros((0, self.length))
        self.X_v2z_train = np.zeros((0, self.length))
        self.X_v2z_test = np.zeros((0, self.length))
        self.X_c1_train = np.zeros((0, self.length))
        self.X_c1_test = np.zeros((0, self.length))
        self.X_c2_train = np.zeros((0, self.length))
        self.X_c2_test = np.zeros((0, self.length))
        self.X_c3_train = np.zeros((0, self.length))
        self.X_c3_test = np.zeros((0, self.length))
        self.X_t1_train = np.zeros((0, self.length))
        self.X_t1_test = np.zeros((0, self.length))
        self.X_t2_train = np.zeros((0, self.length))
        self.X_t2_test = np.zeros((0, self.length))


        for idx, info in enumerate(tqdm(infos)):
            # directory for this file
            fdir = os.path.join(rdir, info[0], info[2])
            self._mkdir(fdir)
            newfile = os.path.join(fdir, os.path.split(filepaths[idx])[1])
            if not os.path.exists(newfile):
                copyfile(filepaths[idx], newfile)

            # the files did not contain headers. Here we create labels based on documentation
            labels = ["Bearing1", "Healthy", "HighSpeed", "Misalignment"]
            index_columns_names = ["Cycle"]
            current_columns = ["Current_" + str(i) for i in range(1, 4)]
            temp_columns = ["Temp_" + str(i) for i in range(1, 3)]
            column_names = ["Cycle", "v1", "v2x", "v2y", "v2z", "t1", "t2", "c1", "c2", "c3"]

            all_data = pd.read_csv(newfile, sep=",", header=None).transpose()
            all_data.columns = column_names

            # self.X.append(time_series)
            self.X_v1.append(all_data["v1"].values)
            self.X_v2x.append(all_data["v2x"].values)
            self.X_v2y.append(all_data["v2y"].values)
            self.X_v2z.append(all_data["v2z"].values)

            idx_last = -(all_data["v1"].values.shape[0] % self.length)
            clips = all_data["v1"].values[:idx_last].reshape(-1, self.length)

            n = clips.shape[0]
            n_split = int(3 * n / 4)
            # self.X_train = np.vstack((self.X_train, clips[:n_split]))
            # self.X_test = np.vstack((self.X_test, clips[n_split:]))
            self.X_v1_train = np.vstack((self.X_v1_train, all_data["v1"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_v1_test = np.vstack((self.X_v1_test, all_data["v1"].values[:idx_last].reshape(-1, self.length)[n_split:]))

            self.X_v2x_train = np.vstack((self.X_v2x_train, all_data["v2x"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_v2x_test = np.vstack((self.X_v2x_test, all_data["v2x"].values[:idx_last].reshape(-1, self.length)[n_split:]))
            self.X_v2y_train = np.vstack((self.X_v2y_train, all_data["v2y"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_v2y_test = np.vstack((self.X_v2y_test, all_data["v2y"].values[:idx_last].reshape(-1, self.length)[n_split:]))
            self.X_v2z_train = np.vstack((self.X_v2z_train, all_data["v2z"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_v2z_test = np.vstack((self.X_v2z_test, all_data["v2z"].values[:idx_last].reshape(-1, self.length)[n_split:]))

            self.X_c1_train = np.vstack((self.X_c1_train, all_data["c1"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_c1_test = np.vstack((self.X_c1_test, all_data["c1"].values[:idx_last].reshape(-1, self.length)[n_split:]))
            self.X_c2_train = np.vstack((self.X_c2_train, all_data["c2"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_c2_test = np.vstack((self.X_c2_test, all_data["c2"].values[:idx_last].reshape(-1, self.length)[n_split:]))
            self.X_c3_train = np.vstack((self.X_c3_train, all_data["c3"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_c3_test = np.vstack((self.X_c3_test, all_data["c3"].values[:idx_last].reshape(-1, self.length)[n_split:]))

            self.X_t1_train = np.vstack((self.X_t1_train, all_data["t1"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_t1_test = np.vstack((self.X_t1_test, all_data["t1"].values[:idx_last].reshape(-1, self.length)[n_split:]))
            self.X_t2_train = np.vstack((self.X_t2_train, all_data["t2"].values[:idx_last].reshape(-1, self.length)[:n_split]))
            self.X_t2_test = np.vstack((self.X_t2_test, all_data["t2"].values[:idx_last].reshape(-1, self.length)[n_split:]))

            label_index = self.labels.index(info[3])
            self.y_train += [label_index] * n_split
            self.y_test += [label_index] * (n - n_split)

            self.files.append(self.labels[label_index])

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_v1_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_v1_train = self.X_v1_train[index]
        self.X_v2x_train = self.X_v2x_train[index]
        self.X_v2y_train = self.X_v2y_train[index]
        self.X_v2z_train = self.X_v2z_train[index]
        self.X_c1_train = self.X_c1_train[index]
        self.X_c2_train = self.X_c2_train[index]
        self.X_c3_train = self.X_c3_train[index]
        self.X_t1_train = self.X_t1_train[index]
        self.X_t2_train = self.X_t2_train[index]
        self.y_train = tuple(self.y_train[i] for i in index)

        # shuffle test samples
        index = list(range(self.X_v1_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_v1_test = self.X_v1_test[index]
        self.X_v2x_test = self.X_v2x_test[index]
        self.X_v2y_test = self.X_v2y_test[index]
        self.X_v2z_test = self.X_v2z_test[index]
        self.X_c1_test = self.X_c1_test[index]
        self.X_c2_test = self.X_c2_test[index]
        self.X_c3_test = self.X_c3_test[index]
        self.X_t1_test = self.X_t1_test[index]
        self.X_t2_test = self.X_t2_test[index]

        self.y_test = tuple(self.y_test[i] for i in index)