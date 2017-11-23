# -*- coding: utf-8 -*-
import gensim
import logging
import os
import numpy as np
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
months = ["january","february", "march","april","may","june","july","august","september","october","november","december"]
wordnumbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
num_topics=20 			
alpha=0.1
eta=0.01

stoplist = set(stopwords.words('english'))
fpath = os.path.join("preprocessed.txt")
with open(fpath, "r") as script:
	filelines =script.readlines()
documents = filelines  
np.random.seed(42)
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda=models.ldamodel.LdaModel.load("model")
X=[]
for l in lda[corpus]:
	t=[0]*lda.num_topics
	for m in l:
		t[m[0]]=m[1]
	X.append(t)
X=np.array(X)
A = np.load("model.expElogbeta.npy")
T=np.dot(X,A)

# doc="Here we introduce a new model of natural textures based on the feature spaces of convolutional neural networks optimised for object recognition. Samples from the model are of high perceptual quality demonstrating the generative power of neural networks trained in a purely discriminative fashion. Within the model, textures are represented by the correlations between feature maps in several layers of the network. We show that across layers the texture representations increasingly capture the statistical properties of natural images while making object information more and more explicit. The model provides a new tool to generate stimuli for neuroscience and might offer insights into the deep representations learned by convolutional neural networks."
# doc="To infer a multilayer representation of high-dimensional count vectors, we propose the Poisson gamma belief network (PGBN) that factorizes each of its layers into the product of a connection weight matrix and the nonnegative real hidden units of the next layer. The PGBN's hidden layers are jointly trained with an upward-downward Gibbs sampler, each iteration of which upward samples Dirichlet distributed connection weight vectors starting from the first layer (bottom data layer), and then downward samples gamma distributed hidden units starting from the top hidden layer. The gamma-negative binomial process combined with a layer-wise training strategy allows the PGBN to infer the width of each layer given a fixed budget on the width of the first layer. The PGBN with a single hidden layer reduces to Poisson factor analysis. Example results on text analysis illustrate interesting relationships between the width of the first layer and the inferred network structure, and demonstrate that the PGBN, whose hidden units are imposed with correlated gamma priors, can add more layers to increase its performance gains over Poisson factor analysis, given the same limit on the width of the first layer. "
# doc ="Multiview representation learning is popular for latent factor analysis. Many existing approaches formulate the multiview representation learning as convex optimization problems, where global optima can be obtained by certain algorithms in polynomial time. However, many evidences have corroborated that heuristic nonconvex approaches also have good empirical computational performance and convergence to the global optima, although there is a lack of theoretical justification. Such a gap between theory and practice motivates us to study a nonconvex formulation for multiview representation learning, which can be efficiently solved by a simple stochastic gradient descent method. By analyzing the dynamics of the algorithm based on diffusion processes, we establish a global rate of convergence to the global optima. Numerical experiments are provided to support our theory."
# doc="We consider the popular problem of sparse empirical risk minimization with linear predictors and a large number of both features and observations. With a convex-concave saddle point objective reformulation, we propose a Doubly Greedy Primal-Dual Coordinate Descent algorithm that is able to exploit sparsity in both primal and dual variables. It enjoys a low cost per iteration and our theoretical analysis shows that it converges linearly with a good iteration complexity, provided that the set of primal variables is sparse. We then extend this algorithm further to leverage active sets. The resulting new algorithm is even faster, and experiments on large-scale Multi-class data sets show that our algorithm achieves up to 30 times speedup on several state-of-the-art optimization methods."
doc = "Adversarial robust poisoning attacks"
tex = [[word for word in doc.lower().split() if word not in stoplist]]
dic = corpora.Dictionary(tex)
cor = [dictionary.doc2bow(text) for text in tex]
T1= [0]*len(dictionary.keys())
for c in cor[0]:
	T1[c[0]]=(c[1]*1.0)/len(dic.keys())
print cor

T1=np.array(T1)
T1=T1.reshape(1,-1)
AT=np.transpose(A)
IAAT=np.linalg.inv(np.dot(A,AT))
X1=np.dot(np.dot(T1,AT),IAAT)
# print X1
x=np.argmax(X1[0])
print np.argmax(X[:,x])



# doc="Here we introduce a new model of natural textures based on the feature spaces of convolutional neural networks optimised for object recognition. Samples from the model are of high perceptual quality demonstrating the generative power of neural networks trained in a purely discriminative fashion. Within the model, textures are represented by the correlations between feature maps in several layers of the network. We show that across layers the texture representations increasingly capture the statistical properties of natural images while making object information more and more explicit. The model provides a new tool to generate stimuli for neuroscience and might offer insights into the deep representations learned by convolutional neural networks."
# doc="To infer a multilayer representation of high-dimensional count vectors, we propose the Poisson gamma belief network (PGBN) that factorizes each of its layers into the product of a connection weight matrix and the nonnegative real hidden units of the next layer. The PGBN's hidden layers are jointly trained with an upward-downward Gibbs sampler, each iteration of which upward samples Dirichlet distributed connection weight vectors starting from the first layer (bottom data layer), and then downward samples gamma distributed hidden units starting from the top hidden layer. The gamma-negative binomial process combined with a layer-wise training strategy allows the PGBN to infer the width of each layer given a fixed budget on the width of the first layer. The PGBN with a single hidden layer reduces to Poisson factor analysis. Example results on text analysis illustrate interesting relationships between the width of the first layer and the inferred network structure, and demonstrate that the PGBN, whose hidden units are imposed with correlated gamma priors, can add more layers to increase its performance gains over Poisson factor analysis, given the same limit on the width of the first layer. "
# doc ="Multiview representation learning is popular for latent factor analysis. Many existing approaches formulate the multiview representation learning as convex optimization problems, where global optima can be obtained by certain algorithms in polynomial time. However, many evidences have corroborated that heuristic nonconvex approaches also have good empirical computational performance and convergence to the global optima, although there is a lack of theoretical justification. Such a gap between theory and practice motivates us to study a nonconvex formulation for multiview representation learning, which can be efficiently solved by a simple stochastic gradient descent method. By analyzing the dynamics of the algorithm based on diffusion processes, we establish a global rate of convergence to the global optima. Numerical experiments are provided to support our theory."
# doc="We consider the popular problem of sparse empirical risk minimization with linear predictors and a large number of both features and observations. With a convex-concave saddle point objective reformulation, we propose a Doubly Greedy Primal-Dual Coordinate Descent algorithm that is able to exploit sparsity in both primal and dual variables. It enjoys a low cost per iteration and our theoretical analysis shows that it converges linearly with a good iteration complexity, provided that the set of primal variables is sparse. We then extend this algorithm further to leverage active sets. The resulting new algorithm is even faster, and experiments on large-scale Multi-class data sets show that our algorithm achieves up to 30 times speedup on several state-of-the-art optimization methods."
tex = [[word for word in doc.lower().split() if word not in stoplist]]
dic = corpora.Dictionary(tex)
cor = [dictionary.doc2bow(text) for text in tex]
S=[]
for c in cor[0]:
	word_i=c[0]
	S.append(A[:,word_i])
S=np.array(S)
x=np.argmax(np.mean(S,axis=0))
# print np.var(S,axis=0)
print np.argmax(X[:,x])
