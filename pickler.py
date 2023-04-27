"""This script makes pickle files out of the trained HMM, the test set,
and the words present in the test set but not in the train set (unknown words)"""

import pickle
import nltk.corpus
from hmm import hmm

corpus = list(nltk.corpus.mac_morpho.tagged_sents())
train_set = corpus[:50397]
test_set = corpus[50397:] 

# Define useful variables
hmm = hmm(train_set)

trainset_types = set(hmm['emission'].columns)
# We can use set difference to get all words that appear in the test set but 
# NOT in the train set

testset_types = set([word for sentence in test_set for word, _ in sentence])
unknown_words = testset_types - trainset_types

# Save variables we're gonna use

with open('hmm.pkl' , 'wb') as file:
    pickle.dump(hmm, file)
with open('test_set.pkl' , 'wb') as file:
    pickle.dump(test_set, file)
with open('unknown_words.pkl' , 'wb') as file:
    pickle.dump(unknown_words, file)