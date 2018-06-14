import unittest

from gensim.models import FastText

from datautils import Dataloader

DATAPATH = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_train.csv'
EMBEDDINGS_PATH = '/Users/vlyalin/Downloads/lenta_lower_100.bin'
BATCH_SIZE = 32

class TestDataloader(unittest.TestCase):
    # TODO: rename
    def test_load_embeddings_is_str(self):
        loader = Dataloader(DATAPATH, EMBEDDINGS_PATH, BATCH_SIZE)
        for batch in loader:
            self.assertEqual(len(batch), BATCH_SIZE)
            break

    # TODO: rename
    def test_load_embessings_is_object(self):
        embeddings = FastText.load_fasttext_format(EMBEDDINGS_PATH)
        loader = Dataloader(DATAPATH, embeddings, BATCH_SIZE)
        for batch in loader:
            self.assertEqual(len(batch), BATCH_SIZE)
            break


if __name__ == '__main__':
    unittest.main()
