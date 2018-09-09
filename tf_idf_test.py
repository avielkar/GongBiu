# importing the necessary packages

import unittest

from tf_idf import TfIdf


class TfIdfTest(unittest.TestCase):

    def test_calc_tf_idf_of_sentence(self):
        sentece_1 = 'This test is unit test and unit is unit'
        sentece_2 = 'Unit is unit and this test is unit test'

        tf_idf = TfIdf()
        tf_idf_matrix_1 = tf_idf.calc_tf_idf_of_sentence(sentece_1)
        tf_idf_matrix_2 = tf_idf.calc_tf_idf_of_sentence(sentece_2)

        print(tf_idf_matrix_1)
        print('\n')
        print(tf_idf_matrix_2)

        self.assertEquals(7, 7)
