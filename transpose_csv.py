import csv
import sys
infile = '/Users/diego/Desktop/knightec/Datasets/Knightec/50/5000/50__60_5000_HighSpeed_34.csv'
outfile = '/Users/diego/Desktop/knightec/Datasets/Knightec/50/5000/50__60_5000_HighSpeed_34_2.csv'

with open(infile) as f:
    reader = csv.reader(f, delimiter=',')
    cols = []
    for row in reader:
        cols.append(row)

with open(outfile, 'wb') as f:
    writer = csv.writer(f)
    for i in range(len(max(cols, key=len))):
        writer.writerow([(c[i] if i<len(c) else '') for c in cols])