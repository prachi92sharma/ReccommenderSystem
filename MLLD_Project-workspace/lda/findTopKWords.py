# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:07:07 2017

@author: abhin
"""
import numpy as np
vocab = {}
wordTopic = None

def buildVocab(vocabFile, numTopics):
    global wordTopic
    global vocab
    count = 0
    for line in open(vocabFile):
        value = line.strip()
        vocab[count] = value
        count += 1
    wordTopic = np.zeros((count,numTopics), dtype = int)    

def buildWordTopicMatrix(wordTopicFile):
    count = 0
    for line in open(wordTopicFile):
        wordTopic[count:] = np.array([int(x) for x in line.split(',')]).reshape(1,-1)
        count += 1
        
def findTopKWords(vocabFile, wordTopicFile, numTopics):
    
    buildVocab(vocabFile,numTopics)
    buildWordTopicMatrix(wordTopicFile)
    for k in range(0,numTopics):
        print("top words for topic"+str(k))
        print([vocab[x] for x in wordTopic[:,k].argsort()[::-1][:20]])

np.set_printoptions(threshold=np.nan)  
findTopKWords('vocab.csv','word-topic.csv',20)
