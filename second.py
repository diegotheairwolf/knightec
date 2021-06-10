import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal as sg


# the files did not contain headers. Here we create labels based on documentation
labels = ['inner_race', 'outer_race', 'stator', 'rotor']
index_columns_names = ["Cycle"]
current_columns =["Current_" + str(i) for i in range(1, 4)]
vibration_columns =["Vibration_" + str(i) for i in range(1, 2)]
temp_columns =["Temp_" + str(i) for i in range(1, 3)]
#column_names = index_columns_names + vibration_columns +  temp_columns + current_columns
column_names = ["Cycle", "v1", "v2x", "v2y", "v2z", "T1", "T2", "c1", "c2", "c3"]
print(column_names)


# load data
all_data = pd.read_csv('./data/Healthy/50_60_5000_Healthy.csv', sep=",", header=None)
print("all_data shape: ", all_data.shape)


all_data = all_data.transpose()
print("all_data shape: ", all_data.shape)
print(all_data.head(5))


all_data.columns = column_names
data_backup = all_data.copy()
print(all_data.head(5))


# shift cycle to start at 0
all_data["Cycle"] = all_data["Cycle"]-data_backup["Cycle"][0]
print(all_data.head(5))


# https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn
sns.lineplot(data=pd.melt(all_data[0:100],["Cycle"]), hue="variable", x="Cycle", y="value")
plt.show()

# https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn
vibration_data = pd.DataFrame({
    'Cycle': all_data["Cycle"],
    'v1' : all_data["v1"],
    'v2x' : all_data["v2x"],
    'v2y' : all_data["v2y"],
    'v2z' : all_data["v2z"],
})
sns.lineplot(data=pd.melt(vibration_data[0:100],["Cycle"]), hue="variable", x="Cycle", y="value")
plt.show()

test = vibration_data.to_numpy()
print(test)