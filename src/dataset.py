import json
import torch
from torch.utils.data import Dataset
from collections import Counter
import itertools

# Define special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

class Vocabulary:
    """
    A class to manage the mapping between words and numerical indices.
    """
    def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]):
        self.word2idx = {word: i for i, word in enumerate(specials)}
        self.idx2word = {i: word for i, word in enumerate(specials)}

        # Start indexing words after the special tokens
        for i, (word, _) in enumerate(counter.most_common(), len(specials)):
            self.word2idx[word] = i
            self.idx2word[i] = word

        self.pad_idx = self.word2idx[PAD_TOKEN]
        self.unk_idx = self.word2idx[UNK_TOKEN]
        self.sos_idx = self.word2idx[SOS_TOKEN]
        self.eos_idx = self.word2idx[EOS_TOKEN]

    def __len__(self):
        return len(self.word2idx)

    def to_indices(self, words):
        """Converts a list of words to a list of indices."""
        return [self.word2idx.get(word, self.unk_idx) for word in words]

    def to_words(self, indices):
        """Converts a list of indices to a list of words."""
        return [self.idx2word.get(idx, UNK_TOKEN) for idx in indices]

class HierarchicalDataset(Dataset):
    """
    A custom PyTorch Dataset to handle our hierarchical JSON data.
    """
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Flatten all sentences to build the vocabulary from the entire dataset
        all_sentences = [s.lower().split() for doc in self.data for p in doc['paragraphs'] for s in p['sentences']]
        word_counts = Counter(itertools.chain(*all_sentences))

        self.vocab = Vocabulary(word_counts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one document, processed and tensorized.
        The document is structured as a list of paragraphs, where each paragraph
        is a list of sentence tensors.
        """
        doc = self.data[idx]

        processed_doc = []
        for paragraph in doc['paragraphs']:
            processed_para = []
            for sentence in paragraph['sentences']:
                words = sentence.lower().split()
                # Add Start-of-Sentence and End-of-Sentence tokens
                indices = [self.vocab.sos_idx] + self.vocab.to_indices(words) + [self.vocab.eos_idx]
                processed_para.append(torch.tensor(indices, dtype=torch.long))
            processed_doc.append(processed_para)

        return processed_doc

def collate_fn(batch):
    """
    A custom collate function. For this proof-of-concept, we will process
    documents one by one (batch size of 1), so this function is simple.
    It just returns the first (and only) item in the batch.
    """
    return batch[0]
