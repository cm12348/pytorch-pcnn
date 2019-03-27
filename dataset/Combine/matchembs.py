import linecache
import numpy as np
import random

words = []
glove_words = []
with open('dataset/Combine/vocab.txt', 'r') as f1:
    for line in f1:
        words.append(line.strip())

f4 = open('dataset/Combine/vector_50d.txt', 'w')

print('Indexing word vectors.')

embeddings_index = {}
with open("dataset/Combine/glove.6B.50d.txt", "r") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        glove_words.append(word)

print('Found %s word vectors.' % len(embeddings_index))


# embeddings_index.values
error = 0
for word in words:
    if word not in glove_words:
        error += 1
        f4.write(word+" ")
        for i in range(50):
            f4.write(str(round(random.random()*2-1,5))+" ")
        f4.write("\n")
    else:
        f4.write(word+" ")
        for element in embeddings_index[word].flat:
            f4.write(str(element)+" ")
        f4.write("\n")
        # f4.write(linecache.getline("glove.6B.50d.txt", glove_words.index(word)))
    # try:
    #     pos = glove_words.index(word)# if word in glove_words else -1
    #     f4.write(linecache.getline("glove.6B.50d.txt", pos))
    # except ValueError, e:
    #     # print(e)
    #     error += 1
f4.close()

print(error, " words not found in glove_words.")