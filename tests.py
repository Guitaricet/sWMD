import unittest

from gensim.models import FastText

from datautils import DataLoader

DATAPATH = '/Users/vlyalin/Downloads/SBER_FAQ/sber_faq_train.csv'
EMBEDDINGS_PATH = '/Users/vlyalin/Downloads/lenta_lower_100.bin'
BATCH_SIZE = 32

class TestDataLoader(unittest.TestCase):
    # TODO: rename
    def test_load_embeddings_is_str(self):
        loader = DataLoader(DATAPATH, EMBEDDINGS_PATH, BATCH_SIZE)
        self.assertEqual(loader.batch_size, BATCH_SIZE)
        for batch in loader:
            self.assertEqual(len(batch), BATCH_SIZE)
            break
        batch = loader.get_random_batch()
        self.assertEqual(len(batch), loader.batch_size)
        test_batch_size = 12
        batch = loader.get_random_batch(batch_size=test_batch_size)
        self.assertEqual(len(batch), test_batch_size)

    # TODO: rename
    def test_load_embessings_is_object(self):
        embeddings = FastText.load_fasttext_format(EMBEDDINGS_PATH)
        loader = DataLoader(DATAPATH, embeddings, BATCH_SIZE)
        for batch in loader:
            self.assertEqual(len(batch), BATCH_SIZE)
            break


if __name__ == '__main__':
    unittest.main()
