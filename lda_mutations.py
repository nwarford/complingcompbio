import numpy as np
from gensim.models import LdaModel
from gensim.matutils import Scipy2Corpus
from scipy import sparse
from gensim.corpora import Dictionary

from gensim.test.utils import common_texts
import pandas as pd
import csv
# Create a corpus from a list of texts
#common_dictionary = Dictionary(common_texts)
#common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
#print(common_texts)

data_toy = np.random.randint(low = 0, high = 100, size = (96, 6), dtype = int)
#for i in range(48):
    #for j in range(3):
        #data[i,j] = 0
        #data[95-i,5-j] = 0
#data = data.T
#print(data)

df = pd.read_csv("merged_counts_indels.tsv", sep="\t", index_col = 0)
#df.drop('INS1', axis = 1)
#df.drop(['INS1','INS2','INS3','INS4','DEL1','DEL2','DEL3','DEL4'], axis = 1)
#print(df.columns)
#print(len(df.index))
whereNA = pd.isnull(df).any(1).nonzero()[0]
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
#print(len(df.index))
data = df.to_numpy(dtype = int)
# print(len(data[0]))
data = data.T
################################
# here is the fix for parsing  #
################################
from gensim.matutils import Dense2Corpus

toy_corpus = Dense2Corpus(data_toy)
corpus = Dense2Corpus(data)

"""


print(data[0].size)

print(data[:5])

df = pd.read_csv("sbs_counts.tsv", sep="\t", index_col = 0, encoding='utf-8')
#print(df)
df = df.astype(int)
data = df.to_numpy()
print(data.dtype)
print(data[4][0])

dictionary = Dictionary(data[:5])

corpus = [dictionary.doc2bow(text) for text in data[:5]]
#print(data)
#corpus = sparse.csc_matrix(data)
# print(corpus)
"""


num_topics = 12
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
np.savetxt("lda_output.csv", tops, delimiter=",")
# print(len(tops[0]))
# print(tops)

#from pprint import pprint
#pprint(top_topics)

POLE_topics = {
    'has_POLE' : {},
    'no_POLE' : {}
}

MSI_topics = {
    'MSI_high' : {},
    'MSI_medium' : {},
    'MSI_low' : {}
}

with open('counts_with_metadata.csv',mode='r') as f :
    reader = csv.reader(f)
    header = reader.__next__()
    index = 0
    for document in corpus :

        while index in whereNA : # need to skip the NAs that were removed
            reader.__next__()
            index += 1

        # This block returns the topic with highest probability for the document as highestTopic
        docTopics = model.get_document_topics(document)
        # print(docTopics)
        probs = [item[1] for item in docTopics]
        highest = probs.index(max(probs))
        topicIndices = [item[0] for item in docTopics]
        highestTopic = topicIndices[highest]
        # print(highestTopic)

        row = reader.__next__()
        # Here, we find out the metadata info - both POLE status and which bucket of MSI it's in
        if row[1545] == 'None' or row[1545] == 'UNKNOWN' : # will throw out later
            # It is in the no_POLE group
            poleGroup = 'no_POLE'
        else :
            poleGroup = 'has_POLE'

        if row[1547] == 'NA' :
            msiGroup = 'MSI_low'
        else :
            MSI_num = int(row[1547])
            if MSI_num <= 20 :
                msiGroup = 'MSI_low'
            elif MSI_num > 20 and MSI_num <= 50 :
                msiGroup = 'MSI_medium'
            else :
                msiGroup = 'MSI_high'

        # Here, we update the appropriate dictionaries
        if highestTopic in POLE_topics[poleGroup] :
            currVal = POLE_topics[poleGroup][highestTopic]
            POLE_topics[poleGroup][highestTopic] = currVal + 1
        else :
            POLE_topics[poleGroup][highestTopic] = 1

        if highestTopic in MSI_topics[msiGroup] :
            currVal = MSI_topics[msiGroup][highestTopic]
            MSI_topics[msiGroup][highestTopic] = currVal + 1
        else :
            MSI_topics[msiGroup][highestTopic] = 1


        index+=1
        # print(model.get_document_topics(document))
print(POLE_topics)
print(MSI_topics)
