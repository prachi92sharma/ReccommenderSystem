import gensim
import logging
import os
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stoplist = set(stopwords.words('english'))
fpath = os.path.join("preprocessed.txt")
with open(fpath, "r") as script:
	filelines =script.readlines()

documents = filelines  

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = models.ldamodel.LdaModel(corpus, num_topics=10)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, alpha=0.01, eta = 0.01,update_every=1, chunksize=100, passes=1000)
lda.save("model")
#print lda.print_topics(10)
lda=models.ldamodel.LdaModel.load("model")
i=0
fp=open("processed_features.txt","w")
for d in documents:
	bow = dictionary.doc2bow(d.split())
	x=""
	x=x+str(i)
	for l in lda.get_document_topics(bow, minimum_probability=0, minimum_phi_value=None, per_word_topics=False):
		x=x+","+str(l[1])
	i=i+1
	fp.write(x+"\n")
fp.close()


