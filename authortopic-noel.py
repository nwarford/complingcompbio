from gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus, Dictionary
# from gensim.test.utils import common_dictionary, datapath, temporary_file
import random
import numpy as np

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

# corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

# First, let's make an array that has 5 entries with 96 potential columns
# basically, we're doing doc2bow manually
# bag_of_words = []
# for i in range(5) :
#     # pick 10 mutations for each genome
#     topic_dict = {}
#     for k in range(10) :
#         mutation = random.randint(0,95)
#         if mutation not in topic_dict :
#             topic_dict[mutation] = 1
#         else :
#             currval = topic_dict[mutation]
#             topic_dict[mutation] = currval + 1
#
#     # make them tuples, like bag of words in gensim expects
#     topic_list = []
#     for entry in topic_dict :
#         tuple = (entry, float(topic_dict[entry]))
#         topic_list.append(tuple)
#
#     bag_of_words.append(topic_list)
#
# # bag_of_words is now a bag-of-words-style corpus in the way gensim expects
# # so now we can generate a dictionary straight from the bag of words, rather than the normal streaming order
# # from https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary.from_corpus
# mutation_dict = Dictionary.from_corpus(bag_of_words)

data = np.random.randint(low = 0, high = 100, size = (96, 6), dtype = int)
for i in range(48):
    for j in range(3):
        data[i,j] = 0
        data[95-i,5-j] = 0
data = data.T
print(data)

dictionary = Dictionary(data)

corpus = [dictionary.doc2bow(text) for text in data]

model = AuthorTopicModel(
    corpus, author2doc=author2doc, id2word=dictionary, num_topics=2
)

# construct vectors for authors
author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

print(author_vecs)
