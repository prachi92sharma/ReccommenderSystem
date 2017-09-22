from string import digits

class NaiveBayesTrain:
	wordMap = {}
	labelMap = {}
	vocabCount = None
	distinctClasses = None
	
	def __init__(self):
		self.vocabCount = 0
		self.distinctClasses = []
	
	def preProcessLine(self, line):
		# Convert uppercase to lowercase
		line = line.strip().lower() 
		# Replace punctuation
		line = line.replace(',','').replace('%','').replace('+','').replace('&','').replace('.','').replace('#','').replace(';','').replace('\'','').replace('$','').replace('-','').replace(':','').replace('"','').replace('@en','').replace('[','').replace(']','').replace('(','').replace(')','').replace('?','').replace("'s",'').replace("\\",'').replace('/','')
		# Remove Digit
		line = line.translate(None, digits)
		return line

	def removeStopWords(self, words):
		# List of Stop Words
		stopWords = ['a','b','c','d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the']
		return [i for i in words if i not in stopWords]

	def updateWordMap(self,labels, tokens):
		for token in tokens:
			if token in self.wordMap:
				valueDict = self.wordMap[token]
				for label in labels:
					if label in valueDict:
						valueDict[label] += 1
					else:
						valueDict[label] = 1
				self.wordMap[token] = valueDict
			else:
				self.vocabCount = self.vocabCount + 1
				valueDict = {}
				for label in labels:
					valueDict[label] = 1
				self.wordMap[token] = valueDict

	def updateLabelMap(self,labels, tokens):
		for label in labels:
			if label in self.labelMap:
				valueList = self.labelMap[label]
				self.labelMap[label] = [valueList[0] + 1, valueList[1] + len(tokens)]
			else:
				self.distinctClasses.append(label)
				valueList = [1,len(tokens)]
				self.labelMap[label] = valueList

	def fit(self, filePath):
		for l in open(filePath):
			labels,line = l.split(' ',1)
			if labels == 'doc_file' or labels == 'label_file' or labels == 'cat_file':
				continue
			line = self.preProcessLine(line).split()
			#print(line)
			tokens =  self.removeStopWords(line[2:])
			#sys.exit(0)
			labels = labels.split(',')
			self.updateWordMap(labels, tokens)
			self.updateLabelMap(labels, tokens)
		return self