import matplotlib.pylab as plt
#matplotlib inline 
#plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', size=12) 

import nltk
#nltk.download()
#nltk.download('punkt')

raw_docs = ["Here are some very simple basic sentences.", "They won't be very interesting, I'm afraid.", "The point of these examples is to _learn how basic text cleaning works_ on *very simple* data."]
            
from nltk. tokenize import word_tokenize
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]
print (tokenized_docs)

import string
print(string.punctuation)

mydoclist = ['Mireia loves me more than Hector loves me', 'Sergio likes me more than Mireia loves me', 'He likes basketball more than footbal']
from collections import Counter
for doc in mydoclist:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    print (tf.items())
    
def build_lexicon(corpus):  # define a set with all possible words included
                            # in all the sentences or "corpus"
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon
def tf(term, document):
    return freq(term, document)
def freq(term, document):
    return document.split().count(term)
vocabulary = build_lexicon(mydoclist)
doc_term_matrix = []
print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
for doc in mydoclist:
    print ('The doc is "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    print ('The tf vector for Document %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string))
    doc_term_matrix.append(tf_vector)
    
print ('All combined, here is our master document term matrix: ')
print (doc_term_matrix)


import math
import numpy as np
def l2_normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalizer(vec))
print ('A regular old document term matrix: ')
print (np.matrix(doc_term_matrix))
print ('\nA document term matrix with row-wise L2 norms of 1:')
print (np.matrix(doc_term_matrix_l2))

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount +=1
    return doccount
def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / (float(df)) )
my_idf_vector = [idf(word, mydoclist) for word in vocabulary]
print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
print ('The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']')


