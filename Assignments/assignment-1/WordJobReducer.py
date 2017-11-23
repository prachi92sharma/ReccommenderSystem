#!/usr/bin/env python

import sys

def reduce(word, labels):
    # Keeping count of number of distinct tokens
    sys.stderr.write("reporter:counter:Vocabulary,TOTAL_VOCAB_COUNT,1\n")
    
    labelMap = {}
    output = []
    for label in labels:
        if label in labelMap:
            labelMap[label] += 1
        else:
            labelMap[label] = 1
    
    for label in labelMap:
        output.append([label,labelMap[label]])

    print '%s\t%s' % (word, output)

currentWord = None
labels = []
for line in sys.stdin:
    line = line.strip()
    word, label = line.split('\t', 1)
    if word == currentWord:
        labels.append(label)
    else:
        if currentWord != None:
            reduce(currentWord,labels)
        currentWord = word
        labels = [label]
if word == currentWord:
    reduce(currentWord,labels)