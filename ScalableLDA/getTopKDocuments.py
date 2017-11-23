
import numpy as np
import sys

vocab = {}
documentTopic = None
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
    for line in open(docFile):
        document.append(line.strip())

def getTopKDocs(numTopics):

    train = documentTopic[:-6,:]
    klDivergenceArr = np.zeros((1,np.shape(train)[0]), dtype = float)
    test = documentTopic[-6:,:]
    query = np.average(test,axis = 0).reshape(1,-1)
    for i in range(0,np.shape(train)[0]):
        klDivergenceArr[0,i] = findKLDivergence(train[i,:].reshape(1,-1) , query)
    
    return np.argsort(klDivergenceArr)[0][:20]

def findKLDivergence(P,Q):
    # P is 1x20 and Q is 1x20
    klDivergence = 0.0
    for i in range(0,np.shape(P)[1]):
        klDivergence += P[0,i] * np.log(P[0,i] / Q[0,i])
    return klDivergence

def findTopKDocuments(docFile,documentTopicFile, numTopics):
    buildDocument(docFile)
    buildDocumentTopicMatrix(documentTopicFile, numTopics)
    indices = getTopKDocs(numTopics)
    
    for i in indices:
        print document[i]
        print '\n'

np.set_printoptions(threshold=np.nan)  
findTopKDocuments('preprocessed.txt','doc-topic-normalized.csv',20)


    #sys.exit(0)
    #queryMat = np.dot(np.ones((np.shape(train)[0],1),dtype=float), query)
    #num = np.sum(train * queryMat, axis = 1).reshape(1,-1)
    #print np.shape(num)
    #den = np.linalg.norm(train,axis = 1).reshape(-1,1) * np.linalg.norm(queryMat,axis = 1).reshape(-1,1)
    #print np.sort(klDivergenceArr)[0]