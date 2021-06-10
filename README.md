# knightec

knightec_py3:
creates a class where all the variables and data are stored. this is done in __init__.py
- generate metadata.txt file that scrapes all of the csv data files in the data folder
- loads only the selected rpm, sample rate files and parses them
- creates slices of shape MxM 

preprocess.py:
- creates an instance of knightec_py3 where data is loaded into
- pickles the data into file to be used later in Google Colab

unpickle.py:
- used as an example/reference on how to unpickle the data
