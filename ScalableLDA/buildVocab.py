from string import digits

vocab = {}
reverseVocab = {}

def buildVocab(fileName):
	global vocab
	global reverseVocab

	count = 0
	docCount = 0
	fp = open('dataFile.csv','w')
	for doc in open(fileName):
		docCount += 1
		tokens = doc.strip().split()
		#print tokens
		
		if tokens == []:
			continue
		
		if tokens[0] == 'chiru':
			tokens = tokens[1:]

		for token in tokens:
			if token not in vocab:
				vocab[token] = count
				reverseVocab[count] = token 
				count += 1
		fp.write(",".join([str(vocab[token]) for token in tokens])+"\n")        
	fp.close()
	return vocab,reverseVocab,docCount