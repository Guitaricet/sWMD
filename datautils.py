import random
import logging

import numpy as np
import pandas as pd

from gensim.models.fasttext import FastText
from pymystem3 import Mystem


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


# from
# https://github.com/akutuzov/universal-pos-tags/blob/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map
RNC_2_UNIPOS = {
    'A': 'ADJ',
    'ADV': 'ADV',
    'ADVPRO': 'ADV',
    'ANUM': 'ADJ',
    'APRO': 'DET',
    'COM': 'ADJ',
    'CONJ': 'SCONJ',
    'INTJ': 'INTJ',
    'NONLEX': 'X',
    'NUM': 'NUM',
    'PART': 'PART',
    'PR': 'ADP',
    'S': 'NOUN',
    'SPRO': 'PRON',
    'UNKN': 'X',
    'V': 'VERB'
}


class DataLoader:
    """
    Iterator

    :param datapath: path to data.csv
    :param embeddings: path to fasttesxt binary embeddings file or gensim.models.FastText object
    :getitem returns: tuple (X, BOW, indices, label) where
        X — numpy array of word embeddings with shape (text_len, embed_dim) where text_len may vary
        BOW - numpy array of word counts with size (n_unique_tokens,)
        ids - numpy array of word indices with size (n_unique_tokens,)
        label - int, class index
    :iterator returns: list of getitem objects with size = batch_size
    BOW and ids are weekly related with each other

    Yes, I know this is a terrible format, and I hope it will be changed in the future
    """
    def __init__(self, datapath, embeddings, batch_size, tokens_with_pos=False, lemmatize=True, frac=1.0):
        self.datapath = datapath
        self.batch_size = batch_size
        self.lemmatize = lemmatize
        self.tokens_with_pos = tokens_with_pos

        if isinstance(embeddings, str):
            self._embeddings = FastText.load_fasttext_format(embeddings)
        elif isinstance(embeddings, FastText):
            self._embeddings = embeddings
        else:
            raise ValueError('embeddings should be either path to fastttext model or gensim.models.FastText object')

        self._m = Mystem()
        self._data = pd.read_csv(datapath, sep='\t', names=['text', 'label'])
        if 0 < frac < 1.0:
            self._data = self._data.sample(int(len(self._data) * frac))

        self._tok2idx = dict()
        self._idx2tok = []
        self._default_token = '<UNK>'
        self._default_embedding = np.random.uniform(-1, 1, self._embeddings.vector_size)

        # Note: speed up a bit, via tokenized sentences cashing?
        # make tok2idx and idx2tok:
        logging.info('Making tok2idx and idx2tok...')
        self._tok2idx[self._default_token] = 0
        self._idx2tok.append(self._default_token)
        for _, row in self._data.iterrows():
            for token in self._tokenize(row['text']):
                if token not in self._tok2idx:
                    self._tok2idx[token] = len(self._tok2idx)
                    self._idx2tok.append(token)
        logging.info('done')

        self._len = len(self._data)
        self._n_batches = self._len // batch_size + int(self._len % batch_size > 0)
        self._batch_pointer = 0

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        data_row = self._data.iloc[i]
        label = int(data_row['label'])
        X, bow, indices = self._preprocess(data_row['text'])
        return X, bow, indices, label

    def __next__(self):
        res = self._make_batch(self._batch_pointer * self.batch_size)
        self._batch_pointer += 1
        return res

    def __iter__(self):
        yield self.__next__()

    @property
    def labels(self):
        return self._data['label'].tolist()

    def reset_batch_pointer(self):
        self._batch_pointer = 0

    def batch_for_indices(self, indices):
        batch = []
        for i in indices:
            batch.append(self.__getitem__(i))
        return batch

    def get_random_batch(self, batch_size=None):
        """
        Return random batch with the structure identical to iterator result
        :param batch_size: optional, if not indicated, uses self.batch_size
        """
        if batch_size is None:
            batch_size = self.batch_size
        ids = random.sample(range(self.__len__()), batch_size)
        return self.batch_for_indices(ids)

    def _make_batch(self, _batch_start):
        batch = []
        for i in range(self.batch_size):
            if _batch_start >= self._len:
                self.reset_batch_pointer()
                raise StopIteration
            batch.append(self.__getitem__(_batch_start + i))
        return batch

    def _tokenize(self, text):
        """
        Tokenize and add pos tags (for rusvectores embeddigns)
        If self.lemmatize, tags are lemmatized
        """
        tokens = []
        processed = self._m.analyze(text)
        for wordinfo in processed:
            if ' ' not in wordinfo['text']:  # mystem returns space symbols as tokens
                token = wordinfo['text']
                try:
                    lemma = wordinfo['analysis'][0]['lex'].lower().strip()
                    pos = wordinfo['analysis'][0]['gr'].split(',')[0]
                    pos = pos.split('=')[0].strip()
                    # convert to Universal Dependencies POS tag format:
                    pos = RNC_2_UNIPOS[pos]
                    if self.lemmatize:
                        _token = lemma
                    else:
                        _token = token.lower()
                    if self.tokens_with_pos:
                        _token = _token + ' ' + pos

                except (KeyError, IndexError):
                    _token = token.lower()
                    if self.tokens_with_pos:
                        _token = _token + '_X'

                tokens.append(_token)

        return tokens

    def _preprocess(self, text):

        bow_big = np.zeros(len(self._tok2idx))

        for token in self._tokenize(text):
            if token in self._tok2idx:
                bow_big[self._tok2idx[token]] += 1
            else:
                bow_big[self._tok2idx[self._default_token]] += 1

        n_unique_tokens = sum(bow_big > 0)
        x = np.zeros([n_unique_tokens, self._embeddings.vector_size])
        bow_small = np.zeros(n_unique_tokens)
        indices = np.zeros(n_unique_tokens, dtype=np.int32)

        pointer = 0
        n_default_embeddings = 0
        for idx, count in enumerate(bow_big):
            if count > 0:
                bow_small[pointer] = count
                indices[pointer] = idx
                token = self._idx2tok[idx]
                if token in self._embeddings:
                    x[pointer, :] = self._embeddings[token]
                else:
                    x[pointer, :] = self._default_embedding
                    n_default_embeddings += 1
                pointer += 1
        
        if 2 * n_default_embeddings > n_unique_tokens:
            logging.warning('Too many default embeddings for text (more than a half of tokens)')

        return x, bow_small, indices
