import torch
from utils import get_device

class Vocab(object):
    def __init__(self):
        super(Vocab, self).__init__()
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_index = 0

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.next_index
            self.index_to_word[self.next_index] = word
            self.next_index += 1
        return self.word_to_index[word]

    def add_context(self, context):
        for word in context:
            self.add_word(word)

    def get_tensor(self, sample):
        return [self.add_word(w) for w in sample]

    def __len__(self):
        return len(self.word_to_index)

    def get_vocab(self):
        return self.word_to_index.keys()


if __name__ == '__main__':
    from data import raw_text
    vocab = Vocab()
    vocab.add_context(raw_text)
    sample = "processes manipulate other abstract things".split()
    print(vocab.get_tensor(sample))
