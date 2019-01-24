import torch
from torch.nn import Embedding, Linear, Module


def get_model(vocab_size, model_type, context_size, embedding_dim):
    if model_type == 'CBOW':
        return CBOW(vocab_size, embedding_dim, context_size)
    elif model_type == 'NGRAM':
        return NGram(vocab_size, embedding_dim, context_size)
    elif model_type == 'SKIPGRAM':
        return SkipGram(vocab_size, embedding_dim, context_size)
    elif model_type == 'NEG':
        return NegativeSampling(vocab_size, embedding_dim, context_size)
    else:
        raise ValueError(f"ERROR: No Model of Type: {model_type}")

class NegativeSampling(Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NegativeSampling, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        center, context = torch.split(inputs, 1, dim=1)
        embeds_cen = self.embeddings(center).view((inputs.size(0), -1)).unsqueeze(1)
        embeds_ctx = self.embeddings(context).view((inputs.size(0), -1)).unsqueeze(1)
        scores = torch.bmm(embeds_cen, embeds_ctx.transpose(1, 2)).squeeze(1)
        return scores


class SkipGram(Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipGram, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.size(0), -1))
        return self.linear(embeds)


class CBOW(Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1).view((inputs.size(0), -1))
        return self.linear(embeds)


class NGram(Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGram, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=1).view((inputs.size(0), -1))
        return self.linear(embeds)
