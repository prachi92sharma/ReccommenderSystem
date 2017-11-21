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
import gensim
import logging
from gensim import corpora, models, similarities


months = ["january","february", "march","april","may","june","july","august","september","october","november","december"]
wordnumbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
word_dict = defaultdict(int)
stopwords_list = set(stopwords.words('english'))

fp = open('query.txt','w')
fpath = os.path.join("dukki.txt")
for line in open("dukki.txt"):
	if line == '\n':
		continue
	words = re.findall(re.compile('\w+'), line.lower().strip())
	wlist = [w for w in words if w not in stopwords_list if len(w) > 2]
	wlist = [w for w in wlist if w not in months ]
	wlist = [w for w in wlist if w not in wordnumbers ]
	for word in wlist:
		word_dict[word] += 1  
	fp.write(' '.join(wlist)+ '\n')
fp.close()
print (len(word_dict))






lda=models.ldamodel.LdaModel.load('lda.model')
dictionary = corpora.Dictionary.load('dictionary.dict')







fpath = os.path.join('query.txt')
with open(fpath, "r") as script:
	filelines =script.readlines()
documents = filelines
texts = [[word for word in document.lower().split()] for document in documents]
#print texts
ques_vec = []
for lis in texts:
	ques_vec = dictionary.doc2bow(lis)
	#ques_vec.append(x)
print ques_vec
topic_vec = []
topic_vec = lda[ques_vec]
print topic_vec
word_count_array = numpy.empty((len(topic_vec), 2), dtype = numpy.object)
for i in range(len(topic_vec)):
	word_count_array[i, 0] = topic_vec[i][0]
	word_count_array[i, 1] = topic_vec[i][1]

idx = numpy.argsort(word_count_array[:, 1])
idx = idx[::-1]
word_count_array = word_count_array[idx]

final = []
final = lda.print_topic(word_count_array[0, 0], 1)

question_topic = final.split('*') ## as format is like "probability * topic"

print question_topic[1]








"""

i=0
fp=open("processed_query.txt","w")
for d in documents:
	bow = dictionary.doc2bow(d.split())
	x=""
	x=x+str(i)
	for l in lda.get_document_topics(bow, minimum_probability=0, minimum_phi_value=None, per_word_topics=False):
		x=x+","+str(l[1])
	i=i+1
	fp.write(x+"\n")
fp.close()
"""
def cosine_similarity(x,y):
	s=0
	for i in range(len(x)):
		s=s+x[i]*y[i]
	return s

files1 = open("processed_features.txt","r")
#files2 = open("processed_query.txt","r")
fi1=[]
#fi2=[]
for f in files1:
	if f:
		fi1.append(f.strip("\n"))
#for f in files2:
#	if f:
#		fi2.append(f.strip("\n"))

train=fi1[0:]
test=question_topic[0]
testlist=[]
for t in test:
	x=[]
	for s in t.split(",")[1:]:
		x.append(float(s))
	testlist.append(x)

avg_vector=[0]*len(testlist[0])
#print testlist
for t in testlist:
	for i in range(len(t)):
		avg_vector[i]=avg_vector[i]+t[i]

avg_vector=[a/len(testlist) for a in avg_vector]
#print avg_vector

cos_array=[]
for t in train:
	x=[]
	for s in t.split(",")[1:]:
		x.append(float(s))
	cos_array.append(cosine_similarity(x,avg_vector))
cos_array=np.array(cos_array)

x=cos_array.argsort()[::-1][:20]
#print cos_array
#print x

files=open("preprocessed.txt","r")
fi=[]
for f in files:
	if f:
		fi.append(f)

#fp=open("output.txt","a")
for t in x:
	#fp.write(fi[t])
	print t,fi[t]
#fp.close()


