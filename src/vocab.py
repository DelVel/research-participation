import os
from collections import Counter

import dill
import nltk
from pycocotools.coco import COCO
from tqdm import tqdm

from src.dataset import train_caption, val_caption


class Vocabulary:
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[unk_token]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_list, threshold_):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for json in json_list:
        coco = COCO(json)
        ids = coco.anns.keys()
        for i, id_ in enumerate(tqdm(ids)):
            caption = str(coco.anns[id_]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold_]

    # Create a vocab wrapper and add some special tokens.
    vocab_ = Vocabulary()
    vocab_.add_word(pad_token)
    vocab_.add_word(start_token)
    vocab_.add_word(end_token)
    vocab_.add_word(unk_token)

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab_.add_word(word)
    return vocab_


def vocab_from_annotations():
    json_list = [train_caption, val_caption]
    vocab_ = build_vocab(json_list=json_list, threshold_=4)
    print(f"Total vocabulary size: {len(vocab_)}")
    os.makedirs(os.path.split(vocab_pkl_path)[0], exist_ok=True)
    with open(vocab_pkl_path, 'wb') as f:
        dill.dump(vocab_, f)
    print(f"Saved the vocabulary wrapper to '{vocab_pkl_path}'")
    return vocab_


def init_vocab():
    if os.path.exists(vocab_pkl_path):
        with open(vocab_pkl_path, 'rb') as f:
            return dill.load(f)
    else:
        return vocab_from_annotations()


pad_token = '<pad>'
start_token = '<start>'
end_token = '<end>'
unk_token = '<unk>'

vocab_pkl_path = './data/vocab.pkl'
vocab = init_vocab()
padding_idx = vocab(pad_token)
end_idx = vocab(end_token)
padding_len = 60
vocab_size = len(vocab)
