import unittest
from bag_of_words import BagOfWords
import numpy


class BagOfWordsTests(unittest.TestCase):

    def test_parse_sentence_to_tuple(self):
        sentence = 'This test is unit test and unit is unit'
        words_list = ['and', 'is', 'test', 'this', 'unit']
        words_count = numpy.asarray([1, 2, 2, 1, 3])

        bag_of_words = BagOfWords()
        [ret_words_list, ret_words_count] = bag_of_words.parse_sentence_to_tuple(sentence)

        self.assertEqual(ret_words_count, words_count)
        self.assertEqual(ret_words_list, words_list)
