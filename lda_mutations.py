import numpy as np
from gensim.models import LdaModel
from gensim.matutils import Scipy2Corpus
from scipy import sparse

data = np.random.randint(low = 0, high = 100, size = (96, 5), dtype = int)
#print(data)
corpus = sparse.csc_matrix(data.any())
print(corpus)
num_topics = 96
chunksize = 20
passes = 20
iterations = 400
eval_every = None
#temp = data[0]  # This is only to "load" the dictionary.                                           
#id2word = data.id2token

model = LdaModel(
    corpus=corpus,
    #id2word=id2word,
    #chunksize=chunksize,
    #alpha='auto',
    #eta='auto',
    #iterations=iterations,
    #num_topics=num_topics,
    #passes=passes,
    #eval_every=eval_every
)

top_topics = model.top_topics(corpus)

#avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
#print('Average topic coherence: %.4f.' % avg_topic_coherence)

#from pprint import pprint
#pprint(top_topics)
