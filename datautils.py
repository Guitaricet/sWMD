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

    Args:
        datapath: path to data.csv
        embeddings: path to fasttesxt binary embeddings file or gensim.models.FastText binary file
        batch_size: int, batch size
        tokens_with_pos: adds POS-tag to tokens

    Returns:
        getitem(): tuple (X, BOW, indices, label) where
            X — numpy array of word embeddings with shape (text_len, embed_dim) where text_len may vary
            BOW - numpy array of word counts with size (n_unique_tokens,)
            ids - numpy array of word indices with size (n_unique_tokens,)
            label - int, class index
            BOW and ids are loosely linked with each other
        iter(): list of getitem objects with size = batch_size
   
    Yes, this is a terrible return format, and I hope it will be changed in the future
    """
    def __init__(self, datapath, embeddings, batch_size, tokens_with_pos=False, lemmatize=True, frac=1.0):
        # TODO: add max_dict_size parameter
        self.datapath = datapath
        self.batch_size = batch_size
        self.lemmatize = lemmatize
        self.tokens_with_pos = tokens_with_pos

        if isinstance(embeddings, str):
            self._embeddings = FastText.load_fasttext_format(embeddings)
        elif isinstance(embeddings, FastText):
            self._embeddings = embeddings
        else:
            raise ValueError('embeddings should be either path to fastttext model or gensim.models.FastText binary file')

        self._m = Mystem()
        self._data = pd.read_csv(datapath, sep='\t', names=['text', 'label'], index_col=False, header=None)
        if 0 < frac < 1.0:
            self._data = self._data.sample(int(len(self._data) * frac))
            self._data.reset_index(inplace=True, drop=True)

        self._tok2idx = dict()
        self._idx2tok = []
        self._default_token = '<UNK>'
        self._default_embedding = np.random.uniform(-1, 1, self._embeddings.vector_size)

        # make tok2idx and idx2tok:
        logging.info('Making tok2idx and idx2tok...')
        tokenized_all = []
        self._tok2idx[self._default_token] = 0
        self._idx2tok.append(self._default_token)
        for _, row in self._data.iterrows():
            tokenized = self._tokenize(row['text'])
            tokenized_all.append(tokenized)
            for token in tokenized:
                if token not in self._tok2idx:
                    self._tok2idx[token] = len(self._tok2idx)
                    self._idx2tok.append(token)
        self._data['tokens'] = tokenized_all
        logging.info('done')

        self._len = len(self._data)
        self._n_batches = self._len // batch_size + int(self._len % batch_size > 0)
        self._batch_pointer = 0

        self._classes = None

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        data_row = self._data.iloc[i]
        label = int(data_row['label'])
        X, bow, indices = self._preprocess(data_row['tokens'])
        return X, bow, indices, label

    def __next__(self):
        res = self._make_batch(self._batch_pointer * self.batch_size)
        self._batch_pointer += 1
        return res

    def __iter__(self):
        yield self.__next__()

    @property
    def labels(self):
        return self._data['label'].as_matrix()
    
    @property
    def classes(self):
        """
        List of unique classes in the data
        """
        if self._classes is None:
            self._classes = self._data['label'].unique()
        return self._classes

    def reset_batch_pointer(self):
        self._batch_pointer = 0

    def batch_for_indices(self, indices):
        batch = []
        for i in indices:
            batch.append(self.__getitem__(i))
        return batch

    def get_random_batch(self, batch_size=None):
        """
        Get random batch with the structure identical to iterator result

        Args:
            batch_size: optional, if not indicated, uses self.batch_size
        """
        if batch_size is None:
            batch_size = self.batch_size
        ids = random.sample(range(self.__len__()), batch_size)
        return self.batch_for_indices(ids)
    
    def sample(self, sample_size):
        """
        Safe indices sampling, guarantees that all classes will be presented in the sample

        Args:
            sample_size: fraction of the whole dataset (0. < sample_size <= 1.)
        
        Returns:
            list of indices of the sample
        """
        assert 0 < sample_size <= 1
        if sample_size == 1:
            return list(range(self.__len__()))

        sample = []
        classes = self.classes
        for c in classes:
            class_indices = self._data[self._data['label'] == c].index
            n_samples = max(1, int(len(class_indices) * sample_size))
            class_sample = np.random.choice(class_indices, n_samples, replace=False)
            sample += list(class_sample)
        np.random.shuffle(sample)
        return sample

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
        if len(tokens) > 0:
            return tokens
        else:
            logging.warning('Zero tokens text!')
            logging.info(text)
            return [' ']

    def _preprocess(self, tokens):

        bow_big = np.zeros(len(self._tok2idx))

        for token in tokens:
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

        assert x.shape[0] > 0
        return x, bow_small, indices
