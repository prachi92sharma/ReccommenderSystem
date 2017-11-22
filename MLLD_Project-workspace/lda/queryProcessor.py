import numpy as np
import random

wordTopic = None
docTopicMatrix = None
vocab = {}
reverseVocab = {}
docWordTopicMatrix = {}
prevTopic = None
document = []

def buildDocumentTopicMatrix(documentTopicFile, numTopics):
    global documentTopic
    count = 0
    documentTopic = np.zeros((len(document),numTopics), dtype = float)  
    for line in open(documentTopicFile):
        documentTopic[count:] = np.array([float(x) for x in line.split(',')]).reshape(1,-1)
        count += 1

def buildDocument(docFile):
    global document
    document = []
    for line in open(docFile):
        #print line
        document.append(line.strip())

def getTopKDocs(numTopics,query):

    train = documentTopic[:]
    klDivergenceArr = np.zeros((1,np.shape(train)[0]), dtype = float)
    for i in range(0,np.shape(train)[0]):
        klDivergenceArr[0,i] = findKLDivergence(train[i,:].reshape(1,-1) , query)
    return np.argsort(klDivergenceArr)[0][:20]

def findKLDivergence(P,Q):
    # P is 1x20 and Q is 1x20
    klDivergence = 0.0
    for i in range(0,np.shape(P)[1]):
        klDivergence += P[0,i] * np.log(P[0,i] / Q[0,i])

    return klDivergence

def findTopKDocuments(docFile,documentTopicFile, numTopics,query):
    buildDocument(docFile)
    buildDocumentTopicMatrix(documentTopicFile, numTopics)
    indices = getTopKDocs(numTopics,query)
    
    for i in indices:
        print document[i]
        print '\n'

def buildVocab(vocabFile, numTopics):
    global wordTopic
    global vocab
    count = 0
    for line in open(vocabFile):
        value = line.strip()
        vocab[count] = value
        reverseVocab[value] = count
        count += 1
    wordTopic = np.zeros((count,numTopics), dtype = int)    

def buildWordTopicMatrix(wordTopicFile):
    global wordTopic
    count = 0
    for line in open(wordTopicFile):
        wordTopic[count:] = np.array([int(x) for x in line.split(',')]).reshape(1,-1)
        count += 1

def resample(wordId, j, numOfTopics):
    topic = 0
    norm = 0
    p = [0 for i in range(0,numOfTopics)]

    for k in range(0,numOfTopics):
        isTopicEqualToWordTopic = 1 if prevTopic[0,j] == k else 0
        count  = 0
        if wordId < np.shape(wordTopic)[0]:
            count = wordTopic[wordId][k]

        a_k = count + 0.01 / (sum(wordTopic[:,k])) + 0.01
        b_k = (docTopicMatrix[0,k] + 0.01) / (sum(docTopicMatrix[0,:]) + numOfTopics * 0.01 -1)
        p_k = a_k * b_k 
        p[k] = p_k
        norm += p_k

    sumUptoK = 0
    r = random.random()
    for k in range(0,numOfTopics):
        sumUptoK += p[k] / norm
        if r < sumUptoK:
            topic = k
            break
    return topic

def flip(new_topic, word, j):
    global prevTopic
    global docTopicMatrix
    global docWordTopicMatrix

    old_topic = prevTopic[0,j]
    if prevTopic[0,j] != new_topic:
        prevTopic[0,j] = new_topic
        docWordTopicMatrix[word][0,old_topic] -= 1
        docTopicMatrix[0,old_topic] -= 1
        docWordTopicMatrix[word][0,new_topic] += 1
        docTopicMatrix[0,new_topic] += 1


def initializeGibbsSampler(numOfTopics):
    global prevTopic
    global docTopicMatrix
    global docWordTopicMatrix

    prevTopic = np.zeros((1,len(docWordTopicMatrix)),dtype = int)
    i = 0
    for word in docWordTopicMatrix.keys():
        randomTopic = random.randint(0,numOfTopics - 1)
        docTopicMatrix[0,randomTopic] += 1
        prevTopic[0,i] = randomTopic
        docWordTopicMatrix[word][0,randomTopic] += 1
        i += 1

def runGibbsSampler(numOfTopics):
    initializeGibbsSampler(numOfTopics)
    for i in range(0,300):
        print i
        j = 0
        for word in docWordTopicMatrix.keys():
            new_topic = resample(word, j, numOfTopics)
            flip(new_topic,word,j)
            j += 1

def getDocumentTopicRepresentation(document, vocabFile, wordTopicFile, numOfTopics):
    global docWordTopicMatrix
    global docTopicMatrix
    buildVocab(vocabFile,numOfTopics)
    buildWordTopicMatrix(wordTopicFile)
    docTopicMatrix = np.zeros((1,numOfTopics), dtype = float)

    words = document.strip().split()
    index = len(vocab)
    for word in words:
        if word in reverseVocab:
            docWordTopicMatrix[reverseVocab[word]] = np.zeros((1,numOfTopics), dtype = int)
        else:
            vocab[index] = word
            reverseVocab[word] = index
            docWordTopicMatrix[reverseVocab[word]] = np.zeros((1,numOfTopics), dtype = int)
            index += 1

    runGibbsSampler(numOfTopics)


document = "uniform spectral hypergraph clustering partitioning tensor methods sampling analysis justified"
getDocumentTopicRepresentation(document, 'vocab.csv','word-topic.csv',20)
docTopicMatrix = (docTopicMatrix + 0.01) / (sum(docTopicMatrix[0,:]) + 20*0.01)
print docTopicMatrix
findTopKDocuments('preprocessed.txt','doc-topic-normalized.csv',20,docTopicMatrix)