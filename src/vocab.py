import os
from collections import Counter

import dill
import nltk
from pycocotools.coco import COCO
from tqdm import tqdm

vocab_pkl_path = './data/vocab.pkl'


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
            return self.word2idx['<unk>']
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
    vocab_.add_word('<pad>')
    vocab_.add_word('<start>')
    vocab_.add_word('<end>')
    vocab_.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab_.add_word(word)
    return vocab_


def vocab_from_annotations():
    annotations_root = 'D:/dataset/coco2014/annotations_trainval2014/annotations'
    json_list = [f'{annotations_root}/captions_train2014.json', f'{annotations_root}/captions_val2014.json']
    vocab_ = build_vocab(json_list=json_list, threshold_=4)
    print(f"Total vocabulary size: {len(vocab_)}")
    with open(vocab_pkl_path, 'wb') as f:
        dill.dump(vocab_, f)
    print(f"Saved the vocabulary wrapper to '{vocab_pkl_path}'")
    return vocab_


def init_vocab():
    global vocab
    if os.path.exists(vocab_pkl_path):
        with open(vocab_pkl_path, 'rb') as f:
            return dill.load(f)
    else:
        return vocab_from_annotations()


vocab = init_vocab()
padding_idx = vocab('<pad>')
padding_len = 60
vocab_size = len(vocab)
