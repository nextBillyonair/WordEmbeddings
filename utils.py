import torch
import matplotlib.pyplot as plt
from collections import Counter
from numpy.random import choice
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    return device

def format(target, model_type, eval=False):
    if model_type == 'NEG':
        target = target.unsqueeze(0).view(target.size(0), -1).float()
    if model_type == 'SKIPGRAM':
        target = target.transpose(0, 1).contiguous().view(1, -1).transpose(0, 1)
    if eval:
        return target
    return target.view(-1)

def build_ngrams(data, context_size=2):
    n_grams = [([data[i + j] for j in range(0, context_size)],
                data[i + context_size]) for i in range(len(data)-context_size)]
    return n_grams

def build_cbow(data, context_size=2):
    cbow = [([data[i + j] for j in range(-context_size, context_size+1) if j != 0],
            data[i]) for i in range(context_size, len(data)-context_size)]
    return cbow

def build_skipgram(data, context_size=2):
    skip = [(data[i], data[i+j]) for i in range(context_size, len(data)-context_size)
            for j in range(-context_size, context_size+1) if j != 0]
    return skip

def build_negsampling(data, context_size=2):
    neg_samples = [((data[i], data[i+j]), 1.0) for i in range(context_size, len(data)-context_size)
            for j in range(-context_size, context_size+1) if j != 0]
    return neg_samples


def build_unigram(data):
    distribution = Counter(data)
    for key in distribution:
        distribution[key] = distribution[key] ** (0.75)
    scale = sum(distribution.values())
    for key in distribution:
        distribution[key] = distribution[key] / scale
    return distribution


def sample_dist(distribution, n=1):
    return choice(list(distribution.keys()), size=n, p=list(distribution.values()))


def parse_data(data, model_type, context_size):
    if model_type == 'CBOW':
        return build_cbow(data, context_size)
    elif model_type == 'NGRAM':
        return build_ngrams(data, context_size)
    elif model_type == 'SKIPGRAM':
        return build_skipgram(data, context_size)
    elif model_type == 'NEG':
        return build_negsampling(data, context_size)
    else:
        raise ValueError(f"ERROR: No Model of Type: {model_type}")


def plot(model, vocab):
    words = list(vocab.get_vocab())
    word_tensor = torch.tensor(vocab.get_tensor(words))
    embeds = PCA(model.embeddings(word_tensor))
    x, y = torch.split(embeds, split_size_or_sections=1, dim=1)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, word in enumerate(words):
        ax.annotate(word, (x[i], y[i]))
    plt.show()


def PCA(X, k=2):
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U,S,V = torch.svd(torch.t(X))
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.size(1))])
    U *= signs
    return torch.mm(X, U[:,:k])


if __name__ == '__main__':
    from data import raw_text
    dist = build_unigram(raw_text)
    print(sample(dist, 5))
    ng = parse_data(raw_text, 'SKIPGRAM', 1)
    for i in range(5):
        print(ng[i])
