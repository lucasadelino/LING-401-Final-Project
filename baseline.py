import nltk
import nltk.corpus
import pickle

lower = [(word.lower(), tag) for (word, tag) in nltk.corpus.mac_morpho.tagged_words()]
data = nltk.ConditionalFreqDist(lower)

with open('hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)

errors = 0
tokens = 0
i = 0

for sent in test_set:
    print(f'Getting sent {i}')
    i += 1
    tokens += len(sent)
    for word, tag in sent:
        word = word.lower()
        pred = max(data[word], key=data[word].get)
        if pred != tag:
            errors += 1

print(f'Most-Frequent-Tag Baseline Accuracy: {((tokens-errors)/tokens)*100:.2f}')