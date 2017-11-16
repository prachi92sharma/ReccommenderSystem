import gensim
import logging
import os
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stoplist = set(stopwords.words('english'))
fpath = os.path.join("/home/prachi.sharma92/Project/abstract1.txt")
with open(fpath, "r") as script:
	filelines =script.readlines()

documents = filelines  

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#lda = LdaModel(corpus, num_topics=10)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1)
lda.print_topics(5)
