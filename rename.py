import os
import re

filepath = os.path.join(os.getcwd(), 'shaft')
files = os.listdir(filepath)
profiles = ["50", "75", "100"]

for j in profiles:
    for i in range(0, 100, 1):
        old_name = "{}/{}__60_5000_Shaft_Bearing_{}.csv".format(filepath, j, i)
        new_name = "{}/{}__60_5000_Shaft+Bearing_{}.csv".format(filepath, j, i)
        try:
            os.rename(old_name, new_name)
        except:
            print("File {} couldn't be found".format(old_name))