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
import h5py

#orig_stdout = sys.stdout
#f = open('out.txt', 'w')
#sys.stdout = f
word_dict = defaultdict(int)
stopwords1 = set(stopwords.words('english'))
fp = open('preprocessed.txt','w')
fpath = os.path.join("/home/prachi.sharma92/Project/abstract1.txt")
for line in open(fpath):
	if line == '\n':
		continue
	#print(line)
	words = re.findall(re.compile('\w+'), line.lower().strip())
	wlist = [w for w in words if not w in stopwords1]
	for word in wlist:
		word_dict[word] += 1  
	print(wlist)
	fp.write(' '.join(wlist)+ '\n')
fp.close()
print (len(word_dict))
np.save("vocab",word_dict)
