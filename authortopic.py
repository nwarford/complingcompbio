from gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus, Dictionary
# from gensim.test.utils import common_dictionary, datapath, temporary_file
import random
import numpy as np
import pandas as pd
# Adapted by Noel from the example at https://radimrehurek.com/gensim/models/atmodel.html

# generate authorship based on a simple rule - author0: 0, 3, 6; author1: 1, 4, 7; author2: 2, 5, 8
author0 = []
author1 = []
author2 = []

for i in range(6) :
    if i % 3 == 0 :
        author0.append(i)
    elif i % 3 == 1 :
        author1.append(i)
    else :
        author2.append(i)

author2doc = {
    'author0': author0,
    'author1': author1,
    'author2': author2
}

#data = np.random.randint(low = 0, high = 100, size = (96, 6), dtype = int)
#for i in range(48):
    #for j in range(3):
        #data[i,j] = 0
        #data[95-i,5-j] = 0
#data = data.T
#print(data)
df = pd.read_csv("sbs_counts.tsv", sep="\t", index_col = 0)

data = df.to_numpy(dtype = int)

dictionary = Dictionary(data)

corpus = [dictionary.doc2bow(text) for text in data]

model = AuthorTopicModel(
    corpus, author2doc=author2doc, id2word=dictionary, num_topics=2
)

# construct vectors for authors
author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

print(author_vecs)
