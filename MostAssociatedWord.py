from typing import List
from gensim.models import KeyedVectors
import gensim.downloader as api

class MostAssociatedWord:

    def __init__(self, words: List, model: str):
        self._words = words
        self._model = KeyedVectors.load(model)
        self._initial_words = []
        self._associated_words = []
        self._avg_distances = []
        self._associated_words_distances_pairing = {}
        self._max_average = 0
        self._most_associated_word = ''
        MostAssociatedWord.initial_words(self)
        MostAssociatedWord.associated_words(self)
        MostAssociatedWord.find_distances(self)
        return

    def initial_words(self) -> List:
        for w in self._words:
            self._initial_words.append(w[0])
        return self._initial_words

    def associated_words(self) -> List:
        for w in self._words:
            self._associated_words.append(w[1])
        return self._associated_words

    def find_distances(self) -> dict:
        for w in self._associated_words:
            distances = self._model.distances(w, self._initial_words)
            self._avg_distances.append(sum(distances) / len(distances))

        self._associated_words_distances_pairing = dict(zip(self._associated_words, self._avg_distances))
        return self._associated_words_distances_pairing

    def most_associated(self) -> str:
        self._min_average = min(self._associated_words_distances_pairing.values())
        for k, v in self._associated_words_distances_pairing.items():
            if v == self._min_average:
                return k

model = api.load("glove-wiki-gigaword-100")
model.save("laks_new_glove.model")








