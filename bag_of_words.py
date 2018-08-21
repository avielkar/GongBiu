from sklearn.feature_extraction.text import CountVectorizer
import numpy


class BagOfWords:
    vectorizer = None  # type: CountVectorizer

    def __init__(self):
        self.vectorizer = CountVectorizer()
        pass

    def parse_sentence_to_tuple(self, sentence: str) -> tuple(list, numpy.array):
        """
        Returns the Bag of words of a sentence.
        :param sentence: The sentence to count it's bag of words.
        :return: The bag of words described by a tuple of words list and there matched array counters.
        """
        x = self.vectorizer.fit_transform(sentence)
        return (self.vectorizer.get_feature_names(), x.toarray())

    def parse_sentence_to_dict(self, sentence: str) -> dict[str, int]:
        """
        The bag of words described by a tuple of words list and there matched array counters.
        :param sentence: The sentence to count it's bag of words.
        :return: The bag of words described by a dictionary with keys as the word and values as the matched occurences of the word.
        """
        [words, words_counts] = self.parse_sentence_to_tuple(sentence)
        bag_of_words_dict = dict()
        index = 0

        for word in words:
            bag_of_words_dict[word] = words_counts[index]
            index += 1

        return bag_of_words_dict
