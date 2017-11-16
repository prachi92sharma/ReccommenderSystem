import gensim
import logging
import os
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stoplist = set(stopwords.words('english'))
fpath = os.path.join("/home/prachi.sharma92/Project/preprocessed.txt")
with open(fpath, "r") as script:
	filelines =script.readlines()

documents = filelines  

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#lda = LdaModel(corpus, num_topics=10)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, alpha=0.1, eta = 0.1,update_every=1, chunksize=100, passes=1)
#lda.print_topics(10)
for files in filelines:
	bow = dictionary.doc2bow(files.split())
	print lda.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
