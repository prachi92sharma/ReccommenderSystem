import gensim
import numpy as np
import logging
import os
from gensim import corpora, models, similarities

def cosine_similarity(x,y):
	s=0
	for i in range(len(x)):
		s=s+x[i]*y[i]
	return s

files = open("processed_features.txt","r")
fi=[]
for f in files:
	if f:
		fi.append(f.strip("\n"))

train=fi[0:-9]
test=fi[-9:]
testlist=[]
for t in test:
	x=[]
	for s in t.split(",")[1:]:
		x.append(float(s))
	testlist.append(x)

avg_vector=[0]*len(testlist[0])
for t in testlist:
	for i in range(len(t)):
		avg_vector[i]=avg_vector[i]+t[i]

avg_vector=[a/len(testlist) for a in avg_vector]

cos_array=[]
for t in train:
	x=[]
	for s in t.split(",")[1:]:
		x.append(float(s))
	cos_array.append(cosine_similarity(x,avg_vector))
cos_array=np.array(cos_array)

print cos_array.argsort()[::-1][:20]