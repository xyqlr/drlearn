import pytest

from rlearn.blackjack import BlackJack
from rlearn.args import args, nnargs

def test_game():
    game = BlackJack()
    state = (['J','K', '2'], ['10', '8'], 1, -1)
    assert  game.get_game_ended(state, 1) == -1

    state=(['4', '7'], ['4', '5'], -1, -1)
    assert  game.get_game_ended(state, -1) == -1

