#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division  # Python 2 users only
import re
from nltk import word_tokenize
import io,sys
import numpy
import os,sys,re,string
import collections
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

months = ["january","february", "march","april","may","june","july","august","september","october","november","december"]
wordnumbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
stemmer = SnowballStemmer("english")
#orig_stdout = sys.stdout
#f = open('out.txt', 'w')
#sys.stdout = f
word_dict = defaultdict(int)
stopwords_list = set(stopwords.words('english'))
fp = open('preprocessed.txt','w')
fpath = os.path.join("abstract1.txt")
for line in open("abstract1.txt"):
	if line == '\n':
		continue
	words = re.findall(re.compile('\w+'), line.lower().strip())
	wlist = [w for w in words if w not in stopwords_list if len(w) > 2]
	wlist = [w for w in wlist if w not in months ]
	wlist = [w for w in wlist if w not in wordnumbers ]
	wlist = [stemmer.stem(w) for w in wlist]
	for word in wlist:
		word_dict[word] += 1  
	fp.write(' '.join(wlist)+ '\n')
fp.close()
#print (stopwords1)
print (len(word_dict))
np.save("vocab",word_dict)
