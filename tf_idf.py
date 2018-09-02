# this file will use the tf-idf ranking method for creating vectors from sentences for pair matching

# importing the library sklearn
from sklearn.feature_extraction.text import TfidfVectorizer



class TfIdf:
    vectorizer - None

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        pass

    def calc_tf_idf_of_sentence(self, sentence: str) -> tuple:
        """
        Returns the tf-idf score of the words in the sentence
        :param sentence: The sentence for the calculation
        :return: The tf-idf score of all the words described by a tuple of words and their matching tf-idf score
        """

        x = self.vectorizer.fit_transform(sentence)
        return x

    def parse_sentence_to_tf_idf_dict(self, sentence):
        """
        The tf-idf described as a dictionary
        :param sentence: The sentence to build
        :return: The resulting tf-idf dictionary
        """

        words, words_tf_idf = self.calc_tf_idf_of_sentence(sentence)
        words_tf_idf_list = words_tf_idf.tolist()
        tf_idf_dict = dict()
        index = 0

        for word in words
            tf_idf_dict[word] = words_tf_idf_list[index]
            index += 1

        return tf_idf_dict