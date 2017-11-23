#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division  # Python 2 users only
import re
from nltk import word_tokenize
import io,sys
from tokenize import tokenize
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


fpath = os.path.join('dukki.txt')
with open(fpath, "r") as script:
	filelines =script.readlines()
testObj = filelines

lda=models.ldamodel.LdaModel.load('lda.model')
dictionary = corpora.Dictionary.load('dictionary.dict')

#findTopic(testObj, dictionary):
text_corpus = []
'''
Preprocess
'''
for query in testObj:
	current_doc = []
	if query == '\n':
		continue
	words = re.findall(re.compile('\w+'), query.lower().strip())
	wlist = [w for w in words if w not in stopwords_list if len(w) > 2]
	wlist = [w for w in wlist if w not in months ]
	wlist = [w for w in wlist if w not in wordnumbers ]
	for word in wlist:
		word_dict[word] += 1
		current_doc.append(word)  
	text_corpus.append(current_doc)
'''
fp = open('prob.txt','w')
def get_doc_topics(lda_model, bow):
	    gamma, _ = lda_model.inference([bow])
	    topic_dist = gamma[0] / sum(gamma[0])
	    return [(topic_id, topic_value) for topic_id, topic_value in enumerate(topic_dist)]

for text in text_corpus:
	bow = dictionary.doc2bow(text)
	#word = ['fun'] # this may need to be a list in list, it's been a while since I've used Gensim so try just a list first
	#bow = model_dictionary.doc2bow(word)
	doc_lda = get_doc_topics(lda, bow)
	dense_vec = gensim.matutils.sparse2full(doc_lda, lda.num_topics)
	fp.write(dense_vec)
	print(dense_vec)

	# method returns all topic distributions regardless of size...(Gensim uses sparse vectors by default)
	# http://stackoverflow.com/questions/17310933/document-topical-distribution-in-gensim-lda
'''	

'''
For each feature vector text, lda[doc_bow] gives the topic
distribution, which can be sorted in descending order to print the 
very first topic
''' 
fp = open('prob.txt','w')
for text in text_corpus:
	doc_bow = dictionary.doc2bow(text)
	#print lda[doc_bow]
	topics = sorted(lda[doc_bow],key=lambda x:x[0])
	print(topics)
	for item in topics:
		fp.write(str(item))
		#print topics[0][0]

i=0
x=""
for item in topics:
	if item[0] == i :
		x=x+","+str(item[1])
	else :
		x=x+","+ str(0)
	i=i+1
fp.write(x+"\n") 
	#fp.write(topics)
	#print(topics[12][0])

fp.close()







