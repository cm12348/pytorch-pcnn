import nltk
from nltk import word_tokenize

words = []
f1 = open('dataset/Combine/sentences.txt', 'r')
f2 = open('dataset/Combine/vocab.txt', 'w')

for line in f1:
    tmp = word_tokenize(line.strip().lower())
    for i in range(len(tmp)):
        words.append(tmp[i])

words = list(set(words))
for i in range(len(words)):
    f2.write(words[i]+'\n')

print('vocab: ', len(words))