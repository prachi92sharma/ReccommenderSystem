import numpy as np
import copy
import random 

class LDAModel(object):
	"""docstring for Model"""
	def __init__(self, arg):
		super(LDAModel, self).__init__(topic_word_count, word_vocab_hash, alpha, beta, num_topics)
		self.nkt=copy.deepcopy(topic_word_count)
		self.n_kt=[] #copy.deepcopy(topic_word_count_new)
		self.nmk=[] #copy.deepcopy(document_topic_count)
		self.vocab=word_vocab_hash.copy()
		self.alpha=alpha
		self.beta=beta
		self.num_topics=num_topics
		self.prev_topics=[]
		self.new_vocab_hash={}

	def get_nkt(self,t,k):
		return self.nkt[k][vocab[t]]

	def get_nmk(self,k,m):
		return self.nmk[m][k]

	def get_n_kt(self,t,k):
		return self.n_kt[k][new_vocab_hash[t]]


	def cumulative_probablity(self,w):
		p=[0]*self.num_topics
		for i in range(self.num_topics):
			a=self.get_nkt(w,i)+self.get_n_kt(w,i)+self.beta
			b=0
			for j in self.vocab.keys():
				b=b+self.get_nkt(j,i)+self.get_n_kt(j,i)+self.beta
			c=self.get_nmk(i,0)+self.alpha
			d=0
			for j in range(self.num_topics):
				d=d+self.get_nmk(j,0)+self.alpha
			p[i]=(a*1.0*c)/(b*(d-1))
		cp=[0]*self.num_topics
		for i in range(self.num_topics):
			j=i+1
			cp[i]=sum(p[:j])
		return cp

	def get_prev_topic_word(self,word):
		return self.prev_topics[vocab[word]]

	def update_prev_topic(self,word,topic):
		self.prev_topics[vocab[word]]=topic

	def gibbs_sample(self,new_document_words):
		doc_words=[]
		z=0
		for word in new_document_words:
			if word in vocab:
				doc_words.append(word)
				if word not in new_vocab_hash:
					self.new_vocab_hash[word]=z
					z=z+1
		self.n_kt=[[0]*len(new_vocab_hash)]*self.num_topics
		self.nmk=[[0]*self.num_topics]
		for word in doc_words:
			j=new_vocab_hash[word]
			u=random.randint(0,self.num_topics-1)
			self.n_kt[u][j]=self.n_kt[u][j]+1
			self.nmk[0][u]=self.nmk[0][u]+1
		self.prev_topics=[-1]*len(self.n_kt[0])
		prev_document_topic=-1
		for z in range(100):
			for word in doc_words:
				cumulative_prob=self.cumulative_probablity(word)			
				u=random.uniform(0,1)
				for p in cumulative_prob:
					if p >= u:
						break
				topic=cumulative_prob.index(p)
				top=self.get_prev_topic_word(word)
				self.update_prev_topic(word,topic)
				self.n_kt[topic][vocab[word]]=self.n_kt[topic][vocab[word]]+1
				if top != -1:
					if self.n_kt[top][vocab[word]]!=0:
						self.n_kt[top][vocab[word]]=self.n_kt[top][vocab[word]]-1
				t=prev_document_topic
				prev_document_topic=topic
				self.nmk[0][topic]=self.nmk[0][topic]+1
				if t != -1:
					if self.nmk[0][t]!=0:
						self.nmk[0][t]=self.nmk[0][t]-1
		v=[0*self.num_topics]
		for i in range(self.num_topics):
			c=self.get_nmk(i,0)+self.alpha
			d=0
			for j in range(self.num_topics):
				d=d+self.get_nmk(j,0)+self.alpha
			v[i]=c*1.0/d
		return v
