from gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus, Dictionary
# from gensim.test.utils import common_dictionary, datapath, temporary_file
import random
import numpy as np
import pandas as pd
# Adapted by Noel from the example at https://radimrehurek.com/gensim/models/atmodel.html

# generate authorship based on a simple rule for the toy data - author0: 0, 3, 6; author1: 1, 4, 7; author2: 2, 5, 8
"""
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
"""

# Now to generate actual authorship

# Mode 1: author 1 is the author who has the POLE mutation, author 2 does not.
has_POLE = []
no_POLE = []



author2doc_POLE = {
    'has_POLE' : has_POLE,
    'no_POLE' : no_POLE
}

# Mode 2: author 1 has MSI-high bucket, author 2 has MSI-medium-bucket, author 3 has MSI-low-bucket

MSI_high = []
MSI_medium = []
MSI_low = []

author2doc_MSI = {
    'MSI_high' : MSI_high,
    'MSI_medium' : MSI_medium,
    'MSI_low' : MSI_low
}

#data = np.random.randint(low = 0, high = 100, size = (96, 6), dtype = int)
#for i in range(48):
    #for j in range(3):
        #data[i,j] = 0
        #data[95-i,5-j] = 0
#data = data.T
#print(data)
df = pd.read_csv("merged_counts_indels.tsv", sep="\t", index_col = 0)

data = df.to_numpy(dtype = int)
print(len(data[0]))
data = data.T
################################
# here is the fix for parsing  #
################################
from gensim.matutils import Dense2Corpus

corpus = Dense2Corpus(data)

# dictionary = Dictionary(data)
# corpus = [dictionary.doc2bow(text) for text in data]

num_topics = 12
curr_author = author2doc_POLE

model = AuthorTopicModel(
    corpus,
    author2doc=curr_author,
    id2word=dictionary,
    num_topics=num_topics
)

# construct vectors for authors
# author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]
#
# print(author_vecs)
tops = model.get_topics()
np.savetxt("authortopic_output.csv", tops, delimiter=",")
print(len(tops[0]))
print(tops)
