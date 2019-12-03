import numpy as np
from gensim.models import LdaModel
from gensim.matutils import Scipy2Corpus
from scipy import sparse
from gensim.corpora import Dictionary

from gensim.test.utils import common_texts
import pandas as pd

# Create a corpus from a list of texts
#common_dictionary = Dictionary(common_texts)
#common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
#print(common_texts)

data_toy = np.random.randint(low = 0, high = 100, size = (96, 6), dtype = int)
print(data_toy[0])
#for i in range(48):
    #for j in range(3):
        #data[i,j] = 0
        #data[95-i,5-j] = 0
#data = data.T
#print(data)
df = pd.read_csv("sbs_counts.tsv", sep="\t", index_col = 0)

data = df.to_numpy(dtype = int)

################################
# here is the fix for parsing  #
################################
from gensim.matutils import Dense2Corpus

toy_corpus = Dense2Corpus(data_toy)
corpus = Dense2Corpus(data)

"""


print(data[0].size)

print(data[:5])

dictionary = Dictionary(data[:5])

corpus = [dictionary.doc2bow(text) for text in data[:5]]
#print(data)
#corpus = sparse.csc_matrix(data)
# print(corpus)
"""


num_topics = 2
chunksize = 2000
passes = 1
iterations = 50
eval_every = 10

#dictionary = Dictionary(data)
#print(dictionary)
#temp = dictionary[0]  # This is only to "load" the dictionary.                                           
#id2word = dictionary.id2token
#id2word = common_dictionary

model = LdaModel(
    corpus=corpus,
    num_topics=num_topics,
    chunksize=chunksize,
    passes=passes,
    iterations=iterations,
    eval_every=eval_every
    #alpha='auto',
    #eta='auto',
)

#top_topics = model.top_topics(corpus)
#print(top_topics)
#print(type(top_topics))

#avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
#print('Average topic coherence: %.4f.' % avg_topic_coherence)
tops = model.get_topics()
print(tops)

#from pprint import pprint
#pprint(top_topics)
