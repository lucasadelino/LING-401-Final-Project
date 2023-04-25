import pickle
import nltk
import viterbi
from concurrent.futures import ThreadPoolExecutor
    
def compare(test_sentence:list, verbose=False)->None:
    """Description"""
    
    words = [word for word, _ in test_sentence]
    tags = [tag for _, tag in test_sentence]
    
    pred_tags = viterbi(words, hmm)
    
    test_length = len(test_sentence)
    errors = []

    for i in range(test_length):
        if pred_tags[i] != tags[i]:
            errors.append((words[i], (pred_tags[i], tags[i])))
    
    accuracy = (test_length - len(errors)) / test_length

    print(f'Input sentence was tagged with {accuracy*100:.2f}% accuracy')

    if verbose:
        if errors: 
            print(f'Tagger made {len(errors)} mistakes:')
            for word, tag_tuple in errors:
                print(f'WORD: {word}')
                print(f'PREDICTED: {tag_tuple[0]}')
                print(f'ACTUAL: {tag_tuple[1]}')

    return errors

# Testing:

# Open files
with open('hmm.pkl', 'rb') as file:
    hmm = pickle.load(file)
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)
with open('unknown_words.pkl' , 'rb') as file:
    unknown_words = pickle.load(file)

# All the errors
errors = []
# How many total tokens were tested
total_tokens = 0

with ThreadPoolExecutor(max_workers=5) as p:
    results = p.map(compare, test_set)

for sent in test_set:
    total_tokens += len(sent)

total_errors = len(errors)
most_common_errors = nltk.FreqDist(errors).most_common()
unkwords_errors = 0

with open('most_common_errors.pkl' , 'wb') as file:
    pickle.dump(most_common_errors, file)

print(f'Total accuracy: {((total_tokens-total_errors)/total_tokens)*100:.2f}%')
print(f'Ratio of errors from unknown words: {(unkwords_errors/total_errors)*100:.2f}%')
