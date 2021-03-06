import unittest
from bag_of_words import BagOfWords
import numpy


class BagOfWordsTests(unittest.TestCase):

    def test_parse_sentence_to_tuple(self):
        sentence = 'This test is unit test and unit is unit'
        words_list = ['and', 'is', 'test', 'this', 'unit']
        words_count = [1, 2, 2, 1, 3]

        bag_of_words = BagOfWords()
        [ret_words_list, ret_words_count] = bag_of_words.parse_sentence_to_tuple(sentence)

        self.assertEqual(ret_words_count.tolist()[0], words_count)
        self.assertEqual(ret_words_list, words_list)

    def test_parse_sentence_to_dict(self):
        sentence = 'This test is unit test and unit is unit'
        words_dict = {'and': 1, 'is': 2, 'test': 2, 'this': 1, 'unit': 3}

        bag_of_words = BagOfWords()
        expected_words_dict = bag_of_words.parse_sentence_to_dict(sentence)

        self.assertEqual(words_dict, expected_words_dict)
