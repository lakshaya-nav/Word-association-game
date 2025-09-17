from most_associated_word import MostAssociatedWord
class AssociationGame:
    def __init__(self, words_players_pairing):
        self._words_players_pairings = words_players_pairing
        return

    def get_winner(self) -> str:
        most_associated = MostAssociatedWord(self._words_players_pairings.values(), 'laks_new_glove.model').most_associated()
        for k, v in self._words_players_pairings.items():
            if most_associated == v[1]:
                return k


game1 = AssociationGame({'Lakshaya': ('phone', 'charger'), 'Nav': ('wire', 'earphones')})
print('The winner is: ', game1.get_winner())
