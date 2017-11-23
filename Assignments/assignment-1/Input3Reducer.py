#!/usr/bin/env python

import sys

		
def reduce(word, classList, docIdArray):
    for docid in docIdArray:
    	print '%s\t%s' % (docid, [word,classList])

currentword = None
classList = "[]"
docIdArray  = []


for line in sys.stdin:
	word, value = line.strip().split('\t', 1)
	if word == currentword:
		if '@' in value:
			docIdArray.append(value)
			currentword = word
		else:
			classList = value
			continue
	
	else:
		if not currentword:
			currentword = word
			if '@' in value:
				docIdArray.append(value)
			else:
				classList = value
		else:
			if docIdArray:
				reduce(currentword, classList, docIdArray)
			
			currentword = word
			docIdArray = []
			classList = "[]"
			if '@' in value:
				docIdArray.append(value)
			else:
				classList = value