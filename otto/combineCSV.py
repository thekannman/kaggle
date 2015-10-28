import csv
import sys
import numpy as np


if len(sys.argv) < 3:
    print "Usage: python combineCSV.py file1.csv file2.csv ..."
    exit(1)

fi = []
for name in sys.argv[1:]:
    fi.append(csv.reader(open(name, "r"), lineterminator='\n'))
fo = csv.writer(open("combined.csv", "w"), lineterminator='\n')

fo.writerow(next(fi[0]))
for reader in fi[1:]:
    next(reader)

probs = [[] for i in range(len(fi))]

for row in fi[0]:
    outrow = [row[0]]
    probs[0] = [float(prob) for prob in row[1:]]
    for i in range (1,len(fi)):
	probs[i] = next(fi[i])[1:]
    for i in range(len(probs)):
	for j in range(len(probs[0])):
	    probs[0][j] += float(probs[i][j])
    probs[0] = [prob/len(probs) for prob in probs[0]]
    outrow.extend(probs[0])
    fo.writerow(outrow)
