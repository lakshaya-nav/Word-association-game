from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import OrderedDict

string = ''
train_iter = list(WikiText2(split='train'))
for i in range(0,10):
    string += train_iter[i]

tokeniser = get_tokenizer('basic_english')


text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

np.random.seed(42)

# Preprocessing the data (tokenisation - splitting text into smaller units, e.g. words)

def tokenise(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def tokenise2(text):
    r = re.compile('[a-z1-9]')
    res = tokeniser(text)
    res = list(filter(r.match, res))
    return res

# creating a map / lookup table for each word and its index, useful later when performing one-hot encoding
def mapping(tokens):
    word_to_id = {}
    id_to_word = {}


    for i, token in enumerate(sorted(set(tokens))):
        word_to_id[token] = i
        id_to_word[i] = token


    return word_to_id, id_to_word


def generate_training_data(tokens, word_to_id, window):  # generates training data, storing inputs in X and ocntext words in y
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))

        )

        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)

def concat(*iterables):  # * allows to pass a variable no. of arguments
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):  # creates the one hot encoding vector for each training data set
    res = [0] * vocab_size
    res[id] = 1
    return res


def init_network(vocab_size, n_embedding):  # initialises two random word embedding weight matrices, n_embedding is no. of features
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)  # transpose of w1
    }
    return model



def forward(model, X, return_cache=True):  # perform all the matrix multiplications
    cache = {}

    cache["a1"] = X @ model["w1"]  # inputs x first word embedding weight matrix
    cache["a2"] = cache["a1"] @ model["w2"]  # output x second word embedding weight matrix
    cache["z"] = softmax(cache["a2"])  # softmax of output

    if not return_cache:
        return cache["z"]
    return cache

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y  # (330x60) - (330x60) = (330x60)
    dw2 = cache["a1"].T @ da2  # (10x330) x (330x60) = (10x60)
    da1 = da2 @ model["w2"].T  # (330x60) x (60x10) = (330x10)
    dw1 = X.T @ da1  # (60x330) x (330x10) = (60x10)
    assert(dw2.shape == model["w2"].shape)  #
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

def cross_entropy(z, y):  # finds the cross entropy loss bewteen predicted output and target output
    return - np.sum(np.log(z) * y)


def get_embedding(model, word):  # get embedding or features for a word
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in vocab")
    one_hot = one_hot_encode(idx, len(word_to_id))
    return forward(model, [one_hot])["a1"]

def get_similarity1(word1, word2):
    vector1 = get_embedding(model, word1)
    vector2 = get_embedding(model, word2)

    cosine = np.dot(vector1[0], vector2[0]) / (norm(vector1) * norm(vector2))
    return cosine

def get_similarity2(model, word1, word2):
    try:
        idx1 = word_to_id[word1]
        idx2 = word_to_id[word2]
    except KeyError:
        print('{} or {} not in vocab'.format(word1, word2))
    one_hot1 = one_hot_encode(idx1, len(word_to_id))
    vector1 = np.asarray(forward(model, [one_hot1])['a1'])
    one_hot2 = one_hot_encode(idx2, len(word_to_id))
    vector2 = np.asarray(forward(model, [one_hot2])['a1'])

    cosine = np.dot(vector1[0], vector2[0]) / (norm(vector1) * norm(vector2))
    return cosine


# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
# plt.style.use("seaborn")

tokens = tokenise(text)
word_to_id, id_to_word = mapping(tokens)
print(word_to_id)
X, y = generate_training_data(tokens, word_to_id, 2)

model = init_network(len(word_to_id), 10)

n_iter = 50  # no. of epochs
learning_rate = 0.05

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

# plt.plot(range(len(history)), history, color="skyblue")
# plt.show()

# learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
# result = forward(model, [learning], return_cache=False)[0]
#
# for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
#     print(word)

print(get_similarity2(model, 'sample', 'data'))


