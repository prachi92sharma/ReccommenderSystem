# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:14:16 2017

@author: abhin
"""

from string import digits
import math
import sys

class NaiveBayesTest:
	model = None
	accuracy = None
	totalLines = 0

	def __init__(self, model):
		self.model = model
		for key in self.model.labelMap:
			self.totalLines += self.model.labelMap[key][0]

	def preProcessLine(self, line):
		# Convert uppercase to lowercase
		line = line.strip().lower()
		# Replace punctuation
		line = line.replace(',','').replace('%','').replace('+','').replace('&','').replace('.','').replace('#','').replace(';','').replace('\'','').replace('$','').replace('-','').replace(':','').replace('"','').replace('@en','').replace('[','').replace(']','').replace('(','').replace(')','').replace('?','').replace("'s",'').replace("\\",'').replace('/','')
		# Remove Digits 
		line = line.translate(None, digits)
		return line

	def removeStopWords(self, words):
		# List of Stop Words

		stopWords = ['a','b','c','d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the']
		return [i for i in words if i not in stopWords]

	def getPredictionAccuracy(self, trueLabels, predictedLabels):
		correct = 0
		for i in range(0, len(trueLabels)):
		    if predictedLabels[i] in trueLabels[i].split(','):
		    	correct += 1
		return float(correct) / len(trueLabels)

	def getConditionalProbability(self, token, label):
		c = 0
		if token in self.model.wordMap and label in self.model.wordMap[token]:
			c = self.model.wordMap[token][label]
		prob = math.log((c + 1.0) / (1.0 + self.model.labelMap[label][1] + self.model.vocabCount))
		return prob

	def getPredictedLabel(self, tokens):
		probArray = []
		for label in self.model.distinctClasses:
			s = 0
			for token in tokens:
				s += self.getConditionalProbability(token, label)
			s = s + math.log(self.model.labelMap[label][0] / float(self.totalLines))
			probArray.append(s)
		return self.model.distinctClasses[probArray.index(max(probArray))]

	def predict(self, filePath):
		trueLabels = []
		predictedLabels = []
		for l in open(filePath):
			labels,line = l.split(' ', 1)
			if labels == 'doc_file:' or labels == 'label_file:' or labels == 'cat_file:':
				continue
			line = self.preProcessLine(line).split()
			tokens =  self.removeStopWords(line[2:])
			trueLabels.append(labels)
			predictedLabels.append(self.getPredictedLabel(tokens))

		return self.getPredictionAccuracy(trueLabels, predictedLabels)