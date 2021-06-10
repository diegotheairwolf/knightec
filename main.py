import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



# the files did not contain headers. Here we create labels based on documentation
labels = ['inner_race', 'outer_race', 'stator', 'rotor']
index_columns_names = ["Cycle"]
current_columns =["Current_" + str(i) for i in range(1, 4)]
vibration_columns =["Vibration_" + str(i) for i in range(1, 2)]
temp_columns =["Temp_" + str(i) for i in range(1, 3)]
#column_names = index_columns_names + current_columns + vibration_columns + temp_columns
column_names = ["Cycle", "v1", "v2x", "v2y", "v2z", "T1", "T2", "c1", "c2", "c3"]
print(column_names)


# load data
# load data
try:
    all_data = pd.read_csv('./data/Healthy/50_60_5000_Healthy.csv', sep=",", header=None)
    print("all_data shape: ", all_data.shape)
except:
    # print >> sys.stderr, "does not exist"
    # print >> sys.stderr, "Exception: %s" % str(e)
    # sys.exit(1)
    print('file not found')

all_data = all_data.transpose()
print("all_data shape: ", all_data.shape)
print(all_data.head(5))
all_data.columns = column_names
print(all_data.head(5))

# plot current data
sns.relplot(data=all_data[0:100], kind="line", x="Cycle", y="Current_1")
plt.show()