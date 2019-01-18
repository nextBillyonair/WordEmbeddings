import torch
from torch.utils.data import Dataset as TorchDataset

from data import raw_text
from vocab import Vocab
from utils import parse_data, get_device, build_unigram, sample_dist

class Dataset(TorchDataset):

    def __init__(self, model_type, context_size):

        samples = []
        targets = []
        self.vocab = Vocab()
        self.vocab.add_context(raw_text)

        pairs = parse_data(raw_text, model_type, context_size)

        if model_type == 'NEG':
            self.unigram_dist = build_unigram(raw_text)
            self.positive_pairs = set(tuple(self.vocab.get_tensor(sample))
                                      for sample, _ in pairs)

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

    def generate_negative_samples(self, pos_samples, K=2):
        pairs = [(item[0][0].item(), item[0][1].item()) for item in torch.split(pos_samples, 1, dim=0)]
        negative_samples = []
        for pair in pairs:
            for _ in range(K):
                negative_samples.append(self.make_sample(pair[0]))
        negative_samples = torch.tensor(negative_samples)
        negative_targets = torch.zeros(pos_samples.size(0)*2)
        return negative_samples, negative_targets

    def make_sample(self, center):
        neg_context = self.vocab.get_tensor(sample_dist(self.unigram_dist))[0]
        new_sample = (center, neg_context)
        if new_sample not in self.positive_pairs:
            return new_sample
        return self.make_sample(center)



if __name__ == '__main__':
    d = Dataset('', 2)
