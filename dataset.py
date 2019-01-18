import torch
from torch.utils.data import Dataset as TorchDataset

from data import raw_text
from vocab import Vocab
from utils import parse_data, get_device, build_unigram

class Dataset(TorchDataset):

    def __init__(self, model_type, context_size):

        samples = []
        targets = []
        self.vocab = Vocab()
        self.vocab.add_context(raw_text)

        if model_type == 'NEG':
            self.unigram_dist = build_unigram(raw_text)

        pairs = parse_data(raw_text, model_type, context_size)
        for sample, target in pairs:
            if model_type == 'NEG':
                samples.append(self.vocab.get_tensor(sample))
                targets.append(target)
            elif model_type == 'SKIPGRAM':
                samples.append(self.vocab.get_tensor([sample]))
                targets.append(self.vocab.get_tensor([target]))
            else:
                samples.append(self.vocab.get_tensor(sample))
                targets.append(self.vocab.get_tensor([target]))

        self.samples = torch.tensor(samples).long()
        self.targets = torch.tensor(targets).long()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        return (sample, target)

    def __len__(self):
        return len(self.samples)



if __name__ == '__main__':
    d = Dataset('', 2)
