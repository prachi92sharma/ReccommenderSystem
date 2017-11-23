#!/usr/bin/env python
import sys

def reduce(label, countArr):
    sys.stderr.write("reporter:counter:Classes, DISTINCT_CLASSES,1")
    print '%s\t%s' % (label, [len(countArr), sum(countArr)])


currentLabel = None
countArr = []
for line in sys.stdin:
    line = line.strip()
    label, count = line.split('\t', 1)
    if label == currentLabel:
        countArr.append(int(count))
    else:
        if currentLabel != None:
            reduce(currentLabel, countArr)
        currentLabel = label
        countArr = [int(count)]
if label == currentLabel:
    reduce(currentLabel, countArr)